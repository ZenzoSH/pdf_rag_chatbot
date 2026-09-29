"""
eval.py – RAGAS evaluation for the PDF chatbot
================================================
Usage:
    uv run eval.py <path_to_pdf> [--questions N] [--out results.csv]

What it does:
  1. Loads and indexes the PDF (same pipeline as main.py / app.py)
  2. Auto-generates N test questions from the document using the Groq LLM
  3. Runs each question through the RAG chain to get answers + retrieved contexts
  4. Evaluates every (question, answer, contexts) triple with RAGAS metrics:
       • Faithfulness       – does the answer only use information from the context?
       • Answer Relevancy   – is the answer relevant to the question?
       • Context Precision  – are the top-ranked chunks actually useful?
       • Context Recall     – does the context cover the answer? (needs ground truth,
                              approximated here from LLM-generated reference answers)
  5. Prints a summary table and saves a detailed CSV
"""

# ── Shim: ragas 0.4.x hard-imports ChatVertexAI which no longer lives in
#    langchain_community — inject a harmless dummy so the import succeeds.
import sys, types

_dummy = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import textwrap
import json
import csv
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from groq import Groq as GroqClient
import asyncio


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RAGAS evaluation for PDF chatbot")
    p.add_argument("pdf", help="Path to the PDF file to evaluate")
    p.add_argument("--questions", type=int, default=5,
                   help="Number of test questions to generate (default: 5)")
    p.add_argument("--out", default=None,
                   help="CSV output file path (default: eval_<timestamp>.csv)")
    return p.parse_args()


# ── Build RAG pipeline (mirrors main.py / app.py) ─────────────────────────────
def build_pipeline(pdf_path: str):
    print("  Loading and indexing PDF…")
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(docs)
    print(f"  → {len(docs)} pages  |  {len(chunks)} chunks")

    embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever    = vector_store.as_retriever()

    llm      = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, return_source_documents=True
    )
    return qa_chain, chunks, llm, embeddings


# ── Auto-generate test questions from the document ────────────────────────────
def generate_questions(chunks, llm, n: int) -> list[str]:
    """
    Picks n evenly-spaced chunks and asks the LLM to produce one crisp,
    self-contained factual question from each chunk.
    """
    print(f"\n[1/3] Generating {n} test questions from the document…")
    step   = max(1, len(chunks) // n)
    sample = chunks[::step][:n]
    questions = []

    for i, chunk in enumerate(sample, 1):
        prompt = textwrap.dedent(f"""
            You are given an excerpt from a document.
            Generate ONE concise, self-contained factual question that can be
            answered using ONLY the text below. Output just the question, nothing else.

            ---
            {chunk.page_content[:800]}
            ---
        """).strip()

        response = llm.invoke(prompt)
        q = response.content.strip().strip('"')
        questions.append(q)
        print(f"  Q{i}: {q}")

    return questions


# ── Generate ground-truth reference answers (for context_recall) ───────────────
def generate_reference_answers(questions: list[str], chunks, llm) -> list[str]:
    """
    Uses the LLM with the full document context to produce reference answers.
    These serve as the ground-truth for context_recall.
    """
    print("\n  Generating reference answers (ground truth)…")
    full_text = "\n\n".join(c.page_content for c in chunks)[:12000]  # cap tokens
    references = []

    for q in questions:
        prompt = textwrap.dedent(f"""
            Using only the document excerpt below, provide a concise, accurate
            answer to the question. Output only the answer.

            Document:
            {full_text}

            Question: {q}
        """).strip()

        resp = llm.invoke(prompt)
        references.append(resp.content.strip())

    return references


# ── Run the RAG chain over all questions ──────────────────────────────────────
def run_rag(qa_chain, questions: list[str]) -> tuple[list[str], list[list[str]]]:
    print("\n[2/3] Running questions through the RAG pipeline…")
    answers  = []
    contexts = []

    for i, q in enumerate(questions, 1):
        result  = qa_chain.invoke(q)
        answer  = result.get("result", "")
        sources = result.get("source_documents", [])
        ctx     = [doc.page_content for doc in sources]

        answers.append(answer)
        contexts.append(ctx)
        print(f"  {i}/{len(questions)} done")

    return answers, contexts


# ── RAGAS evaluation (direct metric scoring — bypasses broken evaluate()) ─────
def run_ragas(
    questions, answers, contexts, references,
    groq_model_name: str,
):
    print("\n[3/3] Running RAGAS evaluation…")

    # ── Build RAGAS LLM (v1: llm_factory with litellm adapter for Groq)
    groq_client = GroqClient(api_key=os.environ["GROQ_API_KEY"])
    ragas_llm   = llm_factory(
        model=groq_model_name,
        provider="groq",
        client=groq_client,
        adapter="litellm",
        temperature=0,
    )

    # ── RAGAS native HuggingFace embeddings
    ragas_embed = RagasHFEmbeddings(model="all-MiniLM-L6-v2")

    # ── Instantiate v1 metrics
    metric_instances = {
        "faithfulness":      Faithfulness(llm=ragas_llm),
        "answer_relevancy":  AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embed),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall":    ContextRecall(llm=ragas_llm),
    }

    # ── Build v1 samples
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=ref,
        )
        for q, a, ctx, ref in zip(questions, answers, contexts, references)
    ]

    # ── Score each sample with each metric directly (bypass evaluate/aevaluate)
    async def _score_all():
        rows = []
        for i, sample in enumerate(samples):
            row = {}
            for name, metric in metric_instances.items():
                try:
                    score = await metric.single_turn_ascore(sample)
                    row[name] = float(score)
                except Exception as e:
                    print(f"    ⚠ {name} failed on Q{i+1}: {e}")
                    row[name] = float("nan")
            rows.append(row)
            print(f"  Scored {i+1}/{len(samples)}")
        return rows

    rows = asyncio.run(_score_all())

    import pandas as pd
    df = pd.DataFrame(rows)
    return df


# ── Pretty-print results ──────────────────────────────────────────────────────
def print_results(df, questions, answers, out_path: str):
    print(f"\n{'═'*52}")
    print("  RAGAS EVALUATION RESULTS")
    print(f"{'═'*52}")

    metric_labels = {
        "faithfulness":      "Faithfulness       ",
        "answer_relevancy":  "Answer Relevancy   ",
        "context_precision": "Context Precision  ",
        "context_recall":    "Context Recall     ",
    }

    scores = {}
    for key, label in metric_labels.items():
        if key in df.columns:
            val = float(df[key].mean())
            scores[key] = val
            bar_len = int(val * 30)
            bar_str = "█" * bar_len + "░" * (30 - bar_len)
            icon    = "✅" if val >= 0.7 else ("⚠️ " if val >= 0.4 else "❌")
            print(f"  {icon} {label} {val:.3f}  [{bar_str}]")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Overall Average: {avg:.3f}")
    print(f"{'═'*52}")

    # Per-question breakdown
    print("\n  Per-question scores:")
    print(f"  {'─'*50}")
    for i, row in df.iterrows():
        q = questions[i]
        print(f"\n  Q{i+1}: {q[:70]}…" if len(q) > 70 else f"\n  Q{i+1}: {q}")
        for key, label in metric_labels.items():
            if key in row and row[key] == row[key]:  # skip NaN
                print(f"    {label}: {row[key]:.3f}")

    # Save CSV
    df["question"] = questions
    df["answer"]   = answers
    df.to_csv(out_path, index=False)
    print(f"\n  📄 Detailed results saved → {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found — '{args.pdf}'")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = args.out or f"eval_{timestamp}.csv"

    print(f"\n{'═'*52}")
    print(f"  PDF CHATBOT — RAGAS EVALUATION")
    print(f"{'═'*52}")
    print(f"  PDF      : {args.pdf}")
    print(f"  Questions: {args.questions}")
    print(f"  Output   : {out_path}")
    print(f"{'═'*52}\n")

    # Build
    qa_chain, chunks, llm, embeddings = build_pipeline(args.pdf)

    # Generate questions + ground truth
    questions  = generate_questions(chunks, llm, args.questions)
    references = generate_reference_answers(questions, chunks, llm)

    # Run RAG
    answers, contexts = run_rag(qa_chain, questions)

    # Evaluate
    result = run_ragas(questions, answers, contexts, references,
                       groq_model_name="openai/gpt-oss-120b")

    # Print + save
    print_results(result, questions, answers, out_path)


if __name__ == "__main__":
    main()

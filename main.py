import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to PDF: ").strip()

    if not os.path.exists(file_path):
        print(f"Error: file not found — '{file_path}'")
        sys.exit(1)

    print("Indexing PDF, please wait...")
    qa_chain = setup_qa_system(file_path)
    print("Ready! Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = qa_chain.invoke(question)

        print("\n── Answer ──────────────────────────────────────")
        print(result["result"])

        sources = result.get("source_documents", [])
        if sources:
            seen_pages = set()
            print("\n── Citations ───────────────────────────────────")
            for doc in sources:
                page = doc.metadata.get("page", 0)
                if page not in seen_pages:
                    seen_pages.add(page)
                    snippet = doc.page_content[:120].replace("\n", " ").strip()
                    print(f"  Page {page + 1}: \"{snippet}...\"")

        print("─" * 49 + "\n")
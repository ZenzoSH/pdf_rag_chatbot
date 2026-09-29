# 📄 PDF RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot for querying PDF documents. Featuring an interactive split-view Streamlit UI with live in-document citation highlighting, a command-line interface (CLI), and an automated benchmark suite powered by RAGAS.

---

## ✨ Features

- **⚡ Fast Vector Retrieval**: Embeds PDF content using HuggingFace's `all-MiniLM-L6-v2` and stores vectors in an in-memory **FAISS** index for instant contextual search.
- **🤖 High-Performance LLM**: Powered by **Groq** (`openai/gpt-oss-120b`) via LangChain for ultra-fast, accurate responses.
- **🖥️ Dual-Pane Streamlit Web App**:
  - **Left Pane**: Interactive chat thread, PDF document metadata (page count, chunk count, chat turns), clear chat, and reset options.
  - **Right Pane**: Embedded PDF viewer with page navigation and automatic text highlighting.
- **📍 Interactive Citations**: Clickable source chips (e.g., `p.3`) beneath bot answers automatically jump to the page and highlight source snippets directly in the PDF document viewer.
- **💻 CLI Support**: Terminal-based chat interface for lightweight, headless environments.
- **📊 Automated RAGAS Benchmark Suite**: Includes `eval.py` to auto-generate test datasets, score model performance across 4 key RAG metrics, and export detailed CSV reports.

---

## 🛠️ Tech Stack

- **Framework**: LangChain (`langchain`, `langchain-community`, `langchain-groq`, `langchain-huggingface`)
- **Web Interface**: Streamlit, `streamlit-pdf-viewer`
- **Embeddings & Vector Store**: HuggingFace (`sentence-transformers`), FAISS (`faiss-cpu`)
- **LLM Provider**: Groq API
- **PDF Engine & Highlighting**: PyMuPDF (`fitz`), PyPDF (`pypdf`)
- **Evaluation**: RAGAS, LiteLLM
- **Package Manager**: `uv`

---

## 🚀 Project Progress & Development Status

### ✅ Progress Completed

- [x] **Core RAG Architecture**:
  - Built document loading, chunking (`RecursiveCharacterTextSplitter`), embedding generation, FAISS indexing, and `RetrievalQA` chain integration.
- [x] **Streamlit Web Application**:
  - Designed custom dark-themed UI (`#0d1117` base with `#6366f1` accenting).
  - Built dual-pane layout separating chat conversation and PDF document view.
  - Integrated `PyMuPDF` annotation engine to search and highlight context snippets when source citations are clicked.
  - Added session state management for uploads, jump-to-page, chat clear, and document reset.
- [x] **CLI Chat Interface**:
  - Created terminal-focused runner (`main.py`) accepting PDF paths, indexing document text, and streaming answers with page-level citations.
- [x] **RAGAS Evaluation Framework**:
  - Developed `eval.py` for end-to-end benchmark testing.
  - Implemented automated test question generation from document chunks.
  - Added synthetic ground-truth reference generation for metric scoring.
  - Integrated 4 core RAGAS metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.
  - Built console report renderer with progress visualization and CSV export functionality.
- [x] **Environment & Package Configuration**:
  - Managed dependencies via `uv` (`pyproject.toml` & `uv.lock`) and `.env` environment loading.

### 🔲 Roadmap & Planned Enhancements

- [ ] **Multi-Document Support**: Query across multiple uploaded PDFs simultaneously.
- [ ] **Persistent Vector Storage**: Save FAISS indexes locally to bypass re-indexing large PDFs on reload.
- [ ] **Hybrid Search**: Combine BM25 keyword matching with FAISS vector similarity for enhanced retrieval precision.
- [ ] **Local Model Support**: Add fallback options for local offline inference using Ollama.

---

## 📦 Installation & Setup

### 1. Prerequisites

- Python `>= 3.14` (or standard Python `3.10+`)
- [uv](https://github.com/astral-sh/uv) (recommended package manager) or standard `pip`
- A [Groq API Key](https://console.groq.com/)

### 2. Clone & Install Dependencies

Using `uv`:
```bash
# Clone repository
git clone https://github.com/ZenzoSH/pdf_rag_chatbot.git
cd pdf_rag_chatbot

# Sync virtual environment & dependencies
uv sync
```

Using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r pyproject.toml
```

### 3. Environment Variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃 Usage

### 1. Streamlit Web Interface (Recommended)

Launch the interactive dual-pane dashboard:

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser. Drag and drop any PDF into the sidebar to start chatting!

### 2. CLI Mode

Run the chatbot directly in your terminal:

```bash
uv run main.py path/to/document.pdf
```

Or run without arguments to enter an interactive prompt:

```bash
uv run main.py
```

### 3. RAGAS Evaluation

Run the automated evaluation benchmark on a PDF:

```bash
uv run eval.py path/to/document.pdf --questions 5 --out evaluation_results.csv
```

**Options**:
- `--questions N`: Number of test questions to auto-generate (default: `5`).
- `--out filepath.csv`: Path to save the detailed evaluation metrics (default: `eval_<timestamp>.csv`).

---

## 📄 License

This project is licensed under the MIT License.

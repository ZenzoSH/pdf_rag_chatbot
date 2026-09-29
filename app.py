import os
import tempfile
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Global ── */
  .stApp               { background: #0d1117; }
  h1,h2,h3,h4          { color: #e6edf3 !important; letter-spacing: -0.02em; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
    padding-top: 8px;
  }

  /* ── Branding ── */
  .brand {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0 14px;
  }
  .brand-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #10b981);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
  }
  .brand-text  { font-size: 1.1rem; font-weight: 700; color: #e6edf3; line-height: 1.2; }
  .brand-sub   { font-size: 0.72rem; color: #484f58; }

  /* ── PDF status badge ── */
  .badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-top: 4px;
  }
  .badge-ready   { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35); color: #10b981; }
  .badge-waiting { background: rgba(245,158,11,.12);  border: 1px solid rgba(245,158,11,.35);  color: #f59e0b; }

  /* ── Stat cards in sidebar ── */
  .stat-row { display: flex; gap: 8px; margin-top: 4px; }
  .stat-card {
    flex: 1; background: #0d1117; border: 1px solid #21262d;
    border-radius: 10px; padding: 8px 12px; text-align: center;
  }
  .stat-num  { font-size: 1.2rem; font-weight: 700; color: #6366f1; }
  .stat-lbl  { font-size: 0.68rem; color: #484f58; margin-top: 1px; }

  /* ── Divider ── */
  .hdivider { border: none; border-top: 1px solid #21262d; margin: 14px 0; }

  /* ── Upload drop zone ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed #30363d; border-radius: 12px;
    padding: 8px; background: #0d1117;
    transition: border-color .2s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #6366f1; }

  /* ── Sidebar buttons ── */
  [data-testid="stSidebar"] .stButton > button {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #8b949e !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    transition: all .15s !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    border-color: #6366f1 !important;
    color: #e6edf3 !important;
  }

  /* ── Citation chips ── */
  .citation-label { font-size: 0.73rem; color: #484f58; margin: 8px 0 4px; }
  .stButton > button[data-testid*="src_"] {
    border-radius: 16px !important;
    border: 1px solid #30363d !important;
    color: #6366f1 !important;
    font-size: 0.76rem !important;
    padding: 2px 10px !important;
    background: transparent !important;
  }
  .stButton > button[data-testid*="src_"]:hover {
    border-color: #6366f1 !important;
    background: rgba(99,102,241,.1) !important;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    border-radius: 14px !important;
    border: 1px solid #21262d !important;
    margin-bottom: 6px !important;
    background: #161b22 !important;
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
  }

  /* ── PDF panel header ── */
  .pdf-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
  }
  .pdf-header-title { font-size: 1rem; font-weight: 600; color: #e6edf3; }
  .pdf-page-badge {
    font-size: 0.75rem; color: #6366f1;
    background: rgba(99,102,241,.1);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 12px; padding: 2px 10px;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center; padding: 60px 20px; color: #484f58;
  }
  .empty-state .icon { font-size: 2.8rem; margin-bottom: 10px; }
  .empty-state p    { font-size: 0.9rem; }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "qa_chain": None,
    "pdf_name": None,
    "pdf_path": None,
    "active_pdf": None,
    "active_page": 0,
    "total_pages": 0,
    "num_chunks": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Backend ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_qa_system(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, return_source_documents=True
    )
    return qa_chain, len(chunks)


def highlight_and_save(pdf_path: str, page_num: int, text: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    hits = page.search_for(text)
    if not hits:
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 5:
                for hit in page.search_for(line):
                    page.add_highlight_annot(hit)
    else:
        for hit in hits:
            page.add_highlight_annot(hit)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc.save(tmp.name)
    doc.close()
    return tmp.name


def render_pdf(pdf_path: str, page_num: int = 0):
    from streamlit_pdf_viewer import pdf_viewer
    pdf_viewer(
        pdf_path,
        width="100%",
        height=790,
        scroll_to_page=page_num + 1,
        render_text=True,
        resolution_boost=2,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">📄</div>
      <div>
        <div class="brand-text">PDF Chatbot</div>
        <div class="brand-sub">Groq · LangChain · HuggingFace</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Indexing…"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

            qa_chain, num_chunks = setup_qa_system(tmp_path)
            doc_info = fitz.open(tmp_path)
            total_pages = doc_info.page_count
            doc_info.close()

            st.session_state.qa_chain   = qa_chain
            st.session_state.pdf_name   = uploaded_file.name
            st.session_state.pdf_path   = tmp_path
            st.session_state.active_pdf = tmp_path
            st.session_state.active_page = 0
            st.session_state.total_pages = total_pages
            st.session_state.num_chunks  = num_chunks
            st.session_state.messages    = []
        st.success("Ready!", icon="✅")

    # Status badge
    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    if st.session_state.pdf_name:
        st.markdown(
            f'<div class="badge badge-ready">✓ &nbsp;{st.session_state.pdf_name}</div>',
            unsafe_allow_html=True,
        )
        # Stats
        st.markdown(f"""
        <div class="stat-row" style="margin-top:10px;">
          <div class="stat-card">
            <div class="stat-num">{st.session_state.total_pages}</div>
            <div class="stat-lbl">Pages</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{st.session_state.num_chunks}</div>
            <div class="stat-lbl">Chunks</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{len(st.session_state.messages)//2}</div>
            <div class="stat-lbl">Turns</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="badge badge-waiting">⏳ &nbsp;No PDF loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    # Page jump (only when PDF loaded)
    if st.session_state.total_pages > 0:
        jump = st.number_input(
            "Jump to page",
            min_value=1,
            max_value=st.session_state.total_pages,
            value=st.session_state.active_page + 1,
            step=1,
            label_visibility="visible",
        )
        if st.button("Go", use_container_width=True):
            st.session_state.active_pdf  = st.session_state.pdf_path
            st.session_state.active_page = int(jump) - 1
            st.rerun()

        st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.active_pdf  = st.session_state.pdf_path
            st.session_state.active_page = 0
            st.rerun()
    with col_b:
        if st.button("📄 Reset PDF", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()


# ── Main split layout ─────────────────────────────────────────────────────────
chat_col, pdf_col = st.columns([11, 13], gap="large")

# ── Left: Chat ────────────────────────────────────────────────────────────────
with chat_col:
    st.markdown("#### 💬 Chat")

    if not st.session_state.messages and not st.session_state.pdf_name:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📂</div>
          <p>Upload a PDF in the sidebar<br>to start chatting.</p>
        </div>
        """, unsafe_allow_html=True)

    # Render history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Citation chips
            if msg.get("sources"):
                st.markdown('<div class="citation-label">Sources</div>', unsafe_allow_html=True)
                # Deduplicate by page
                seen = {}
                for src in msg["sources"]:
                    p = src.metadata.get("page", 0)
                    if p not in seen:
                        seen[p] = src

                chip_cols = st.columns(min(len(seen), 4))
                for j, (page_num, src) in enumerate(seen.items()):
                    with chip_cols[j % 4]:
                        if st.button(f"p.{page_num+1}", key=f"src_{i}_{j}"):
                            with st.spinner("Highlighting…"):
                                annot = highlight_and_save(
                                    st.session_state.pdf_path, page_num, src.page_content
                                )
                            st.session_state.active_pdf  = annot
                            st.session_state.active_page = page_num
                            st.rerun()

    # Input
    if prompt := st.chat_input(
        "Ask a question…",
        disabled=st.session_state.qa_chain is None,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking…"):
                result  = st.session_state.qa_chain.invoke(prompt)
                answer  = result.get("result", str(result))
                sources = result.get("source_documents", [])
            placeholder.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()


# ── Right: PDF viewer ─────────────────────────────────────────────────────────
with pdf_col:
    if st.session_state.active_pdf:
        cur_page = st.session_state.active_page + 1
        total    = st.session_state.total_pages
        st.markdown(
            f'<div class="pdf-header">'
            f'<span class="pdf-header-title">📑 Document Viewer</span>'
            f'<span class="pdf-page-badge">Page {cur_page} / {total}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        render_pdf(st.session_state.active_pdf, st.session_state.active_page)
    else:
        st.markdown("#### 📑 Document Viewer")
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📋</div>
          <p>Your PDF will appear here<br>after you upload one.</p>
        </div>
        """, unsafe_allow_html=True)

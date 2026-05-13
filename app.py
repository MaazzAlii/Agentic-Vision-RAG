"""
==================================================
  AGENTIC RAG - Streamlit Web UI
==================================================
  Run with:  streamlit run app.py
==================================================
"""

import streamlit as st
import time
import json
import os
from pathlib import Path

# ── Page Config ─────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ───────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .tool-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #64b5f6;
        border: 1px solid #1565c0;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px 4px;
        font-family: monospace;
    }
    .kb-badge  { background: #1b3a2f; color: #69f0ae; border-color: #2e7d52; }
    .hist-badge{ background: #3a2828; color: #ff8a80; border-color: #7f0000; }
    .stats-box {
        background: #1a1a2e;
        border: 1px solid #2c2c54;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 13px;
    }
    .chunk-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ───────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False


# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Agentic RAG")
    st.caption("Vision-based chunking + Tool-calling Agent")

    st.divider()

    # Backend selection
    st.subheader("🔧 LLM Backend")
    backend = st.radio(
        "Choose backend:",
        ["ollama", "mistral"],
        help="Ollama = local (free). Mistral = free API (no credit card needed)."
    )
    os.environ["LLM_BACKEND"] = backend

    if backend == "mistral":
        api_key = st.text_input("Mistral API Key", type="password",
                                placeholder="Get free key at console.mistral.ai")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

    if backend == "ollama":
        st.info("🖥️ Make sure Ollama is running:\n```\nollama serve\n```\nModels needed:\n- llava\n- nomic-embed-text\n- mistral")

    st.divider()

    # PDF Upload + Ingestion
    st.subheader("📄 Document Ingestion")
    uploaded_pdf = st.file_uploader("Upload any PDF", type=["pdf"])

    if uploaded_pdf:
        import tempfile
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success(f"📁 Loaded: {uploaded_pdf.name}")

        if st.button("🚀 Ingest PDF", type="primary", use_container_width=True):
            from config import LLM_BACKEND as cfg_backend
            with st.spinner("⏳ Running ingestion pipeline..."):
                progress = st.progress(0, "Starting...")
                try:
                    from ingestion import pdf_to_images, process_batches, save_chunks_to_json, store_in_chromadb

                    progress.progress(10, "📸 Converting PDF to images...")
                    images = pdf_to_images(pdf_path)

                    progress.progress(30, f"🔍 Analyzing {len(images)} pages with Vision LLM...")
                    chunks = process_batches(images)

                    progress.progress(75, "💾 Saving JSON metadata...")
                    save_chunks_to_json(chunks)

                    progress.progress(90, "🗄️ Storing in ChromaDB...")
                    store_in_chromadb(chunks)

                    progress.progress(100, "✅ Done!")
                    st.session_state.ingested = True
                    st.success(f"✅ {len(chunks)} chunks stored in ChromaDB!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                    st.exception(e)

    st.divider()

    # Agent Init
    st.subheader("🤖 Agent")
    if st.button("Initialize Agent", use_container_width=True):
        try:
            from agent import AgenticRAG
            st.session_state.agent = AgenticRAG()
            stats = st.session_state.agent.get_collection_stats()
            st.success(f"✅ Agent ready! {stats['count']} chunks in DB")
        except Exception as e:
            st.error(f"❌ {e}")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.reset()
        st.rerun()

    # Show DB stats
    if st.session_state.agent:
        stats = st.session_state.agent.get_collection_stats()
        st.markdown(f"""
<div class="stats-box">
📊 <b>ChromaDB Status</b><br>
Collection: <code>{stats['collection']}</code><br>
Chunks stored: <b>{stats['count']}</b><br>
Status: {'🟢 Ready' if stats['status']=='ready' else '🔴 Empty'}
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption("Agentic RAG | Vision + Tool Calling")


# ── Main Area ────────────────────────────────────
st.title("🤖 Agentic RAG — Chat Interface")

# Tabs
tab_chat, tab_chunks, tab_explain = st.tabs(
    ["💬 Chat", "📋 Chunk Viewer", "📖 How It Works"]
)

# ── TAB 1: Chat ──────────────────────────────────
with tab_chat:
    if not st.session_state.agent:
        st.warning("⚠️ Initialize the agent from the sidebar first.")
    else:
        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tools_used"):
                    badges = ""
                    for t in msg["tools_used"]:
                        cls = "kb-badge"   if t["tool"] == "search_knowledge_base" else "hist-badge"
                        label = "🔍 KB Search" if t["tool"] == "search_knowledge_base" else "📜 History Check"
                        badges += f'<span class="tool-badge {cls}">{label}</span>'
                    st.markdown(
                        f'<div style="margin-top:6px">{badges}</div>',
                        unsafe_allow_html=True
                    )

        # Chat input
        user_input = st.chat_input("Ask anything about your document...")
        if user_input:
            # Show user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Agent is thinking..."):
                    try:
                        answer, tools_used = st.session_state.agent.chat(user_input)
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                        tools_used = []

                st.markdown(answer)

                # Show which tools were used
                if tools_used:
                    badges = ""
                    for t in tools_used:
                        cls   = "kb-badge"     if t["tool"] == "search_knowledge_base" else "hist-badge"
                        label = "🔍 KB Search" if t["tool"] == "search_knowledge_base" else "📜 History Check"
                        query = t["args"].get("query", t["args"].get("question", ""))
                        badges += f'<span class="tool-badge {cls}">{label}: {query[:40]}</span>'
                    st.markdown(
                        f'<div style="margin-top:8px; font-size:12px">Tools used: {badges}</div>',
                        unsafe_allow_html=True
                    )

            st.session_state.messages.append({
                "role"       : "assistant",
                "content"    : answer,
                "tools_used" : tools_used
            })


# ── TAB 2: Chunk Viewer ──────────────────────────
with tab_chunks:
    st.subheader("📋 Ingested Chunks (chunks.json)")

    if Path("chunks.json").exists():
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        st.info(f"Total chunks: **{len(chunks)}**")

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            search_heading = st.text_input("🔎 Filter by heading keyword")
        with col2:
            page_filter = st.number_input("Filter by page number (0 = all)", min_value=0)

        filtered = chunks
        if search_heading:
            filtered = [c for c in filtered if search_heading.lower() in c["heading"].lower()]
        if page_filter > 0:
            filtered = [c for c in filtered if c["page_no"] == page_filter]

        for c in filtered:
            with st.expander(
                f"Chunk {c['chunk_no']} | Page {c['page_no']} | Batch {c['batch_no']} — {c['heading']}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Page:** {c['page_no']}")
                    st.markdown(f"**Batch:** {c['batch_no']}")
                    st.markdown(f"**Type:** `{c['page_type']}`")
                    st.markdown(f"**Keywords:** {', '.join(c.get('keywords', []))}")
                with col_b:
                    st.markdown(f"**Summary:**\n{c['summary']}")
                st.divider()
                st.markdown(f"**Content (Markdown):**\n\n{c['content'][:1000]}...")
    else:
        st.info("No chunks found. Upload and ingest a PDF first.")


# ── TAB 3: Explanation ───────────────────────────
with tab_explain:
    st.subheader("📖 How This Agentic RAG Works")

    st.markdown("""
## 🏗️ Architecture

```
PDF
 │
 ▼
[pdf2image]
 │  Convert every page to image (DPI=150)
 │
 ▼
[Batch: 5 pages at a time]
 │  Pages 1-5 → Batch 1
 │  Pages 6-10 → Batch 2 ... etc
 │
 ▼
[Vision LLM per page]  ← LLaVA (Ollama) or Pixtral (Mistral)
 │  Returns: heading, sub_headings, content_markdown,
 │           summary, keywords, page_type
 │
 ▼
[chunks.json]  ← All metadata saved here for inspection
 │
 ▼
[Embedding Model]  ← nomic-embed-text (Ollama) or mistral-embed (Mistral)
 │  Each chunk's text → Vector (list of numbers)
 │
 ▼
[ChromaDB]  ← Stores vectors + text + metadata on disk
```

---

## 🤖 Agentic Generation Pipeline

```
User Question
      │
      ▼
[LLM Agent with Tools]  ← mistral (Ollama) or mistral-small (Mistral API)
      │
      │  AGENT DECIDES:
      ├─ Does it need document info? → CALL search_knowledge_base()
      ├─ Is it a follow-up question? → CALL check_chat_history()
      ├─ Needs both?                 → CALL BOTH tools
      └─ General greeting?          → Answer directly (no tools)
      │
      ▼
[Tool Results returned to Agent]
      │
      ▼
[Final Answer] ← Agent synthesizes tools results + generates response
```

---

## ❓ What's the difference between Agentic and Generative model?

| Component | Model | Role |
|-----------|-------|------|
| **Vision (Generative)** | LLaVA / Pixtral | Reads page images, extracts text & metadata |
| **Embedding** | nomic-embed-text / mistral-embed | Converts text to vectors for similarity search |
| **Agent (Agentic LLM)** | Mistral / mistral-small | Decides which tools to call, generates final answer |
| **ChromaDB** | — | Vector database, stores & retrieves chunks |

---

## 🔧 Chunking Strategy

This project uses **Vision-based chunking** (LLM chunking):

- Each **page** becomes one chunk (not arbitrary token splits)
- The Vision LLM reads the **image** of the page
- It identifies **headings, sub-headings, content type**
- Metadata (heading, page_no, keywords) is attached to every chunk
- This is **much better** than simple text splitting because:
  - It understands tables, figures, layouts
  - Works on any PDF (scanned or text-based)
  - Headings are extracted semantically, not by font size heuristics
""")

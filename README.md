# 🤖 Vision Agentic RAG

**A production-grade Agentic RAG system** that reads ANY PDF using Vision AI, stores knowledge in ChromaDB, and answers questions using a tool-calling agent.

> Built with: Python · Streamlit · ChromaDB · Mistral AI · Ollama

---

## ✨ Key Features

- 📄 **Works on any PDF** — scanned or text-based, any language
- 👁️ **Vision-based chunking** — LLM reads page images, not raw text
- 📦 **Batch processing** — pages processed in batches of 5
- 🤖 **Agentic pipeline** — agent decides which tools to call per query
- 🔧 **Tool calling** — searches knowledge base OR checks chat history
- 🗄️ **ChromaDB** — persistent vector storage with full metadata
- 🔀 **Dual backend** — switch between Mistral AI and Ollama in one click

---

## 🏗️ Architecture

```
INGESTION PIPELINE
──────────────────
PDF → Page Images → Batches of 5 → Vision LLM (per page)
    → JSON metadata (heading, summary, keywords, page_no)
    → Embeddings → ChromaDB

AGENTIC GENERATION PIPELINE
────────────────────────────
User Question → Agent LLM (with tools)
    → Tool 1: search_knowledge_base() → ChromaDB search
    → Tool 2: check_chat_history()    → conversation context
    → Final Answer (with page citations)
```

---

## 🚀 Setup — Step by Step

### Step 1: Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/vision-agentic-rag.git
cd vision-agentic-rag
```

### Step 2: Create a virtual environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Python packages
```bash
pip install -r requirements.txt
```

### Step 4: Install Poppler (required by pdf2image)

**Windows:**
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract zip to `C:\poppler\`
3. Add `C:\poppler\Library\bin` to System PATH
4. Restart your terminal

**Linux/Ubuntu:**
```bash
sudo apt install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

---

## 🔧 Choose Your Backend

### Option A — Mistral AI (Recommended, Free API)

1. Get a free API key at: https://console.mistral.ai/
2. No credit card needed
3. Create a file called `api.env` in the project folder:
```
MISTRAL_API_KEY=your_key_here
LLM_BACKEND=mistral
```

### Option B — Ollama (Local, 100% Offline, Free)

1. Download and install Ollama from: https://ollama.com
2. Open a terminal and run:
```bash
ollama serve
```
3. Pull the required models (run each once):
```bash
ollama pull llava              # Vision model — reads PDF images
ollama pull nomic-embed-text   # Embedding model
ollama pull mistral            # Agent / chat model
```
4. Create `api.env`:
```
LLM_BACKEND=ollama
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open your browser at: http://localhost:8501

---

## 💬 How to Use

1. **Select backend** from sidebar (Mistral or Ollama)
2. **Enter API key** if using Mistral (shown in sidebar)
3. **Upload a PDF** using the file uploader in sidebar
4. **Click "Ingest PDF"** — wait for all pages to process
5. **Click "Initialize Agent"**
6. **Go to Chat tab** and start asking questions!

---

## 📁 Project Structure

```
vision-agentic-rag/
├── config.py          → Settings (models, paths, batch size)
├── ingestion.py       → PDF → Images → Vision LLM → ChromaDB
├── agent.py           → Agentic pipeline with tool calling
├── mistral_client.py  → Mistral API calls (no SDK needed)
├── app.py             → Streamlit Web UI
├── main.py            → CLI interface
├── requirements.txt   → Python dependencies
├── api.env            → Your API key (NOT pushed to GitHub)
└── .gitignore         → Keeps api.env and chroma_db private
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Web UI | Streamlit |
| Vision LLM | Mistral Pixtral / LLaVA (Ollama) |
| Embedding | mistral-embed / nomic-embed-text |
| Agent LLM | mistral-small / mistral (Ollama) |
| Vector DB | ChromaDB |
| PDF → Images | pdf2image + Poppler |

---

## ❓ FAQ

**Q: Do I need a GPU?**
For Mistral AI (API): No, runs on any machine.
For Ollama: A GPU helps but is not required. CPU works, just slower.

**Q: Does it work on scanned PDFs?**
Yes — because we convert pages to images first, so it works regardless of whether the PDF has a text layer.

**Q: The ingestion is slow — is that normal?**
Yes. Each page makes one API call to the Vision LLM. A 20-page PDF takes about 25 seconds with Mistral API (due to 1 RPS rate limit).

**Q: Can I use a different PDF each time?**
Yes — just upload a new PDF and click Ingest PDF again. It clears the old data automatically.


---

## 👨‍💻 Author

**Maaz** — CS Student @ NUML Islamabad  
GitHub: github.com/maazzalii

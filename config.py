import os
from dotenv import load_dotenv

# ── Load api.env file automatically ────────────
load_dotenv("api.env")

# ── Choose Backend ──────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "mistral")   # default is now mistral

# ── Ollama Settings (not needed, kept as backup) ──
OLLAMA_BASE_URL     = "http://localhost:11434"
OLLAMA_VISION_MODEL = "llava"
OLLAMA_EMBED_MODEL  = "nomic-embed-text"
OLLAMA_CHAT_MODEL   = "mistral"

# ── Mistral AI Settings ─────────────────────────
MISTRAL_API_KEY      = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_VISION_MODEL = "pixtral-12b-2409"
MISTRAL_CHAT_MODEL   = "mistral-small-latest"
MISTRAL_EMBED_MODEL  = "mistral-embed"

# ── ChromaDB Settings ───────────────────────────
CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "agentic_rag"

# ── Ingestion Settings ──────────────────────────
BATCH_SIZE  = 5
PDF_DPI     = 150
OUTPUT_JSON = "chunks.json"

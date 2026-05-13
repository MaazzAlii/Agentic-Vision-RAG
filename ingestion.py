"""
==================================================
  AGENTIC RAG - Ingestion Pipeline
==================================================
  FLOW:
    1. PDF  →  List of page images (pdf2image)
    2. Images  →  Batches of 5 pages
    3. Each page image  →  Vision LLM
       Vision LLM returns: heading, chunk_no, page_no, markdown content
    4. All chunks  →  Saved to chunks.json
    5. Chunk text  →  Embedded  →  Stored in ChromaDB
==================================================
"""

import os
import io
import json
import base64
import time
import chromadb
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from config import (
    LLM_BACKEND, BATCH_SIZE, PDF_DPI, OUTPUT_JSON,
    CHROMA_PATH, CHROMA_COLLECTION,
    OLLAMA_VISION_MODEL, OLLAMA_EMBED_MODEL,
    MISTRAL_API_KEY, MISTRAL_VISION_MODEL, MISTRAL_EMBED_MODEL
)


# ─────────────────────────────────────────────────────────────────
# STEP 1: PDF → Images
# ─────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str) -> list:
    """Convert every page of a PDF into a PIL Image object."""
    print(f"📄 Loading PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=PDF_DPI)
    print(f"   → {len(images)} pages found")
    return images


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string (for sending to LLM)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────
# STEP 2: Vision LLM → Structured Metadata per Page
# ─────────────────────────────────────────────────────────────────

VISION_PROMPT = """You are a document analysis expert. Look at this page image carefully and extract all information.

Return ONLY a valid JSON object — no extra text, no markdown fences. Use this exact structure:
{
  "heading": "<main heading or title of this page, or 'No Heading' if none>",
  "sub_headings": ["<sub-heading 1>", "<sub-heading 2>"],
  "content_markdown": "<full text content of this page formatted as markdown>",
  "summary": "<1-2 sentence summary of what this page is about>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "page_type": "<type: introduction / content / conclusion / table / figure / other>"
}"""


def extract_page_metadata_ollama(image: Image.Image) -> dict:
    """Use local Ollama vision model (LLaVA) to extract metadata from a page image."""
    import ollama
    img_b64 = image_to_base64(image)
    response = ollama.chat(
        model=OLLAMA_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": VISION_PROMPT,
            "images": [img_b64]
        }]
    )
    return _parse_llm_json(response["message"]["content"])


def extract_page_metadata_mistral(image: Image.Image) -> dict:
    """Use Mistral AI Pixtral (vision) to extract metadata from a page image."""
    from mistral_client import vision_chat
    img_b64 = image_to_base64(image)
    result = vision_chat(MISTRAL_VISION_MODEL, img_b64, VISION_PROMPT)
    return _parse_llm_json(result)


def _parse_llm_json(raw: str) -> dict:
    """Safely parse JSON from LLM output (removes markdown fences if present)."""
    raw = raw.strip()
    # Remove ```json ... ``` fences if LLM added them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])  # remove first and last line
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw as content
        return {
            "heading": "Unknown",
            "sub_headings": [],
            "content_markdown": raw,
            "summary": "Content extracted from page.",
            "keywords": [],
            "page_type": "content"
        }


def extract_page_metadata(image: Image.Image) -> dict:
    """Route to correct vision LLM based on config."""
    if LLM_BACKEND == "ollama":
        return extract_page_metadata_ollama(image)
    else:
        return extract_page_metadata_mistral(image)


# ─────────────────────────────────────────────────────────────────
# STEP 3: Process Batches
# ─────────────────────────────────────────────────────────────────

def process_batches(images: list) -> list:
    """
    Process images in batches of BATCH_SIZE (default 5).
    Each page gets:
      - chunk_id, chunk_no, page_no, batch_no
      - heading, content, metadata from vision LLM
    """
    all_chunks = []
    total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(images))
        batch = images[start:end]

        print(f"\n📦 Batch {batch_idx + 1}/{total_batches}  (pages {start+1}–{end})")

        for i, image in enumerate(batch):
            page_no  = start + i + 1
            chunk_no = len(all_chunks) + 1

            print(f"   🔍 Analyzing page {page_no}...")
            metadata = extract_page_metadata(image)

            chunk = {
                "chunk_id"   : f"chunk_{chunk_no:04d}",
                "chunk_no"   : chunk_no,
                "page_no"    : page_no,
                "batch_no"   : batch_idx + 1,
                "heading"    : metadata.get("heading", "Unknown"),
                "sub_headings": metadata.get("sub_headings", []),
                "content"    : metadata.get("content_markdown", ""),
                "summary"    : metadata.get("summary", ""),
                "keywords"   : metadata.get("keywords", []),
                "page_type"  : metadata.get("page_type", "content"),
            }
            all_chunks.append(chunk)
            print(f"   ✅ Chunk {chunk_no} | Heading: '{chunk['heading']}'")

            # Rate limit protection for Mistral free API (1 RPS limit)
            # Ollama is local so no delay needed
            if LLM_BACKEND == "mistral":
                time.sleep(1.2)  # slightly over 1 second to be safe

    return all_chunks


# ─────────────────────────────────────────────────────────────────
# STEP 4: Save to JSON
# ─────────────────────────────────────────────────────────────────

def save_chunks_to_json(chunks: list, path: str = OUTPUT_JSON):
    """Save all chunk metadata to a JSON file for inspection."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(chunks)} chunks → {path}")


# ─────────────────────────────────────────────────────────────────
# STEP 5: Embeddings + ChromaDB Storage
# ─────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """Generate embedding vector for a text string."""
    if LLM_BACKEND == "ollama":
        import ollama
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    else:
        from mistral_client import get_embedding as mistral_embed
        return mistral_embed(MISTRAL_EMBED_MODEL, text)


def store_in_chromadb(chunks: list):
    """
    Embed each chunk's content and store in ChromaDB with metadata.
    ChromaDB stores: vector + document text + metadata (page_no, heading, etc.)
    """
    print(f"\n🗄️  Storing in ChromaDB...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)

    # Reset collection for fresh ingestion
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("   → Cleared old collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"description": "Agentic RAG document chunks"}
    )

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for chunk in chunks:
        text = chunk.get("content", "").strip()
        if not text:
            continue  # Skip empty pages

        print(f"   🔢 Embedding chunk {chunk['chunk_no']} (page {chunk['page_no']})...")
        emb = get_embedding(text)

        ids.append(chunk["chunk_id"])
        embeddings.append(emb)
        documents.append(text)
        metadatas.append({
            "chunk_no"  : chunk["chunk_no"],
            "page_no"   : chunk["page_no"],
            "batch_no"  : chunk["batch_no"],
            "heading"   : chunk["heading"],
            "summary"   : chunk["summary"],
            "keywords"  : ", ".join(chunk.get("keywords", [])),
            "page_type" : chunk["page_type"],
        })

    # Batch add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"   ✅ Stored {len(ids)} chunks in ChromaDB at '{CHROMA_PATH}'")


# ─────────────────────────────────────────────────────────────────
# MAIN INGESTION FUNCTION
# ─────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> list:
    """
    Full ingestion pipeline:
      PDF → Images → Batches → Vision LLM → JSON → ChromaDB
    Returns list of all chunk dicts.
    """
    print("=" * 55)
    print("  AGENTIC RAG — INGESTION PIPELINE")
    print("=" * 55)

    # 1. Convert PDF to images
    images = pdf_to_images(pdf_path)

    # 2 & 3. Process in batches, extract metadata via LLM
    chunks = process_batches(images)

    # 4. Save to JSON file
    save_chunks_to_json(chunks, OUTPUT_JSON)

    # 5. Embed and store in ChromaDB
    store_in_chromadb(chunks)

    print("\n" + "=" * 55)
    print(f"  ✅ INGESTION COMPLETE: {len(chunks)} chunks ready")
    print("=" * 55)
    return chunks


# ─────────────────────────────────────────────────────────────────
# Utility: View saved metadata
# ─────────────────────────────────────────────────────────────────

def view_chunks(json_path: str = OUTPUT_JSON):
    """Print a summary of all saved chunks from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"\n📋 Chunks in {json_path}:")
    print(f"{'Chunk':>6} {'Page':>5} {'Batch':>6}  Heading")
    print("-" * 60)
    for c in chunks:
        print(f"  {c['chunk_no']:>4}   {c['page_no']:>4}   {c['batch_no']:>4}  {c['heading'][:45]}")
    print(f"\nTotal: {len(chunks)} chunks")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])
        view_chunks()
    else:
        print("Usage: python ingestion.py <path_to_pdf>")

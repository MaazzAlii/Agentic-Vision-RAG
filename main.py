"""
==================================================
  AGENTIC RAG - Command Line Interface
==================================================
  For quick testing without the web UI.

  Usage:
    # Ingest a PDF first:
    python main.py ingest my_document.pdf

    # Then start chatting:
    python main.py chat

    # Or do both at once:
    python main.py ingest my_document.pdf --chat
==================================================
"""

import sys
import argparse
from pathlib import Path


def run_ingest(pdf_path: str):
    """Run the ingestion pipeline on a PDF file."""
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n📄 Ingesting: {pdf_path}")
    from ingestion import ingest_pdf, view_chunks
    chunks = ingest_pdf(pdf_path)
    view_chunks()
    return chunks


def run_chat():
    """Start interactive CLI chat with the agent."""
    from agent import AgenticRAG

    agent = AgenticRAG()
    stats = agent.get_collection_stats()

    if stats["status"] == "empty":
        print("⚠️  Knowledge base is empty. Ingest a PDF first:")
        print("   python main.py ingest <your_pdf>")
        return

    print(f"\n{'='*55}")
    print("  🤖 AGENTIC RAG — Interactive Chat")
    print(f"  📚 Knowledge base: {stats['count']} chunks loaded")
    print(f"  Backend: {stats}")
    print(f"{'='*55}")
    print("  Commands: 'reset' to clear history | 'quit' to exit")
    print(f"{'='*55}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 Chat history cleared.\n")
            continue

        # Get answer from agent
        print("\n🤔 Thinking...\n")
        answer, tools_used = agent.chat(user_input)

        # Show tools used
        if tools_used:
            for t in tools_used:
                name  = t["tool"]
                query = t["args"].get("query", t["args"].get("question", ""))
                icon  = "🔍" if name == "search_knowledge_base" else "📜"
                print(f"{icon} Tool called: {name}({query[:50]})")
            print()

        print(f"Assistant: {answer}\n")
        print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG — Vision-based PDF Q&A with Tool-Calling Agent"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into ChromaDB")
    ingest_parser.add_argument("pdf", help="Path to PDF file")
    ingest_parser.add_argument("--chat", action="store_true",
                               help="Start chat immediately after ingestion")

    # chat command
    subparsers.add_parser("chat", help="Start interactive chat (requires prior ingestion)")

    # view command
    subparsers.add_parser("view", help="View saved chunks from chunks.json")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args.pdf)
        if args.chat:
            run_chat()

    elif args.command == "chat":
        run_chat()

    elif args.command == "view":
        from ingestion import view_chunks
        view_chunks()

    else:
        parser.print_help()
        print("\n📌 Quick Start:")
        print("  1. python main.py ingest my_document.pdf")
        print("  2. python main.py chat")
        print("  3. Or run the Web UI:  streamlit run app.py")


if __name__ == "__main__":
    main()

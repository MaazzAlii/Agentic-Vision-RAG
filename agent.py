"""
==================================================
  AGENTIC RAG - Agentic Generation Pipeline
==================================================
  WHAT IS "AGENTIC" here?
  ─────────────────────────────────────────────
  A normal RAG always searches the database.
  An AGENTIC RAG decides by itself:
    → "Can I answer from the conversation history?"
    → "Do I need to search the knowledge base?"
    → "Do I need BOTH?"
  It uses TOOLS to do this, just like a human
  assistant would decide what to look up.

  TOOLS AVAILABLE TO THE AGENT:
    1. search_knowledge_base(query, n_results)
       → Searches ChromaDB for relevant document chunks
    2. check_chat_history(question)
       → Checks if this was already discussed in this session
==================================================
"""

import json
import chromadb
from config import (
    LLM_BACKEND, CHROMA_PATH, CHROMA_COLLECTION,
    OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL,
    MISTRAL_API_KEY, MISTRAL_CHAT_MODEL, MISTRAL_EMBED_MODEL
)


# ─────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS  (what the agent can use)
# ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the document knowledge base (ChromaDB) for chunks relevant "
                "to the user's question. Use this when the question asks about content "
                "from the uploaded document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Refined search query to find relevant document chunks"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "How many chunks to retrieve (default: 3, max: 6)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_chat_history",
            "description": (
                "Check the conversation history to see if this question was already "
                "answered or discussed earlier in this session. Use this when the user "
                "says 'earlier', 'before', 'previously', 'what did I ask', or refers "
                "to something from a previous turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or topic to look for in chat history"
                    }
                },
                "required": ["question"]
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """You are a strict document-only Agentic RAG assistant. You ONLY answer questions about the uploaded document.

## YOUR TOOLS:
1.  — searches the document stored in ChromaDB
2.  — checks if this was already discussed in our conversation

## DECISION RULES:
- Question is about document content → ALWAYS use  first
- User refers to "earlier", "before", "you said", "previous" → use  FIRST
- Follow-up question about document → use BOTH tools
- Greeting ("hi", "hello") → reply briefly, do not use tools
- ANY question NOT about the document → REFUSE politely (see refusal rule below)

## STRICT REFUSAL RULE:
If the question is not about the document content (e.g. general knowledge, personal advice,
celebrities, politics, how-to guides unrelated to the document), you MUST respond ONLY with:
"I can only answer questions about the uploaded document. Your question is outside the document scope."
Do NOT attempt to answer off-topic questions under any circumstances.

## PAGE NUMBER RULES — CRITICAL:
- ONLY cite page numbers that actually exist in the retrieved chunks from the tool results
- NEVER guess or invent page numbers
- If the tool returns chunk from page 5, cite (Page 5) — do not change it
- If no page number is in the tool result, do NOT add one
- If information not found in tools: say exactly "This information was not found in the document"
- NEVER say a page number higher than what the tool results show

## RESPONSE RULES:
- Only use information from tool results — never from your own training knowledge
- Always cite exact page numbers from tool results: (Page 3) or (Page 5, 7)
- Be concise and factual
- Never make up, hallucinate, or infer information not present in the chunks"""


# ─────────────────────────────────────────────────────────────────
# AGENTIC RAG CLASS
# ─────────────────────────────────────────────────────────────────

class AgenticRAG:
    """
    Agentic RAG pipeline.

    How it works:
    1. User sends a question
    2. LLM agent decides which tools to call (search KB, check history, or both)
    3. Tools are executed → results returned to agent
    4. Agent generates final answer using tool results
    5. Conversation history is maintained across turns
    """

    def __init__(self):
        self.chat_history: list = []  # Full conversation history
        self.tool_call_log: list = []  # Log of all tool calls (for UI display)

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
            print(f"✅ Connected to ChromaDB collection '{CHROMA_COLLECTION}'")
        except Exception:
            self.collection = None
            print("⚠️  ChromaDB collection not found. Please ingest a PDF first.")

    # ── TOOL IMPLEMENTATIONS ────────────────────────────────────

    def search_knowledge_base(self, query: str, n_results: int = 3) -> str:
        """
        TOOL 1: Embed the query and retrieve top-N chunks from ChromaDB.
        Returns formatted context with page numbers and headings.
        """
        if self.collection is None:
            return "❌ Knowledge base is empty. Please upload and process a PDF first."

        n_results = min(n_results, 6)

        # Embed the query
        query_embedding = self._get_embedding(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return "No relevant chunks found in the knowledge base for this query."

        # Format results
        parts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = round(1 - dist, 2)
            parts.append(
                f"📄 [Page {meta['page_no']}] — {meta['heading']}\n"
                f"Relevance: {similarity} | Keywords: {meta.get('keywords','')}\n"
                f"Summary: {meta.get('summary','')}\n\n"
                f"{doc[:800]}{'...' if len(doc)>800 else ''}"
            )

        return "\n\n---\n\n".join(parts)

    def check_chat_history(self, question: str) -> str:
        """
        TOOL 2: Return recent conversation history as context.
        Agent can use this to answer follow-up questions without re-searching the DB.
        """
        if not self.chat_history:
            return "No conversation history yet. This is the first message."

        # Return last 6 messages (3 exchanges)
        recent = self.chat_history[-6:]
        history_lines = []
        for msg in recent:
            role    = "USER" if msg["role"] == "user" else "ASSISTANT"
            content = msg["content"]
            # Truncate long messages for context window efficiency
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"[{role}]: {content}")

        return "Recent conversation history:\n\n" + "\n\n".join(history_lines)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool by name and return result as string."""
        self.tool_call_log.append({"tool": tool_name, "args": tool_args})

        if tool_name == "search_knowledge_base":
            return self.search_knowledge_base(
                query=tool_args["query"],
                n_results=tool_args.get("n_results", 3)
            )
        elif tool_name == "check_chat_history":
            return self.check_chat_history(tool_args["question"])
        else:
            return f"Unknown tool: {tool_name}"

    # ── EMBEDDING ───────────────────────────────────────────────

    def _get_embedding(self, text: str) -> list:
        """Get embedding vector for retrieval."""
        if LLM_BACKEND == "ollama":
            import ollama
            resp = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
            return resp["embedding"]
        else:
            from mistral_client import get_embedding as mistral_embed
            return mistral_embed(MISTRAL_EMBED_MODEL, text)

    # ── AGENT LOOPS ─────────────────────────────────────────────

    def _ollama_agent_loop(self, messages: list) -> str:
        """
        Ollama agentic loop:
        - Sends messages to LLM with tool definitions
        - If LLM calls a tool → executes it → adds result → continues
        - Stops when LLM gives a final text answer (no more tool calls)
        - Max 5 tool calls to prevent infinite loops
        """
        import ollama
        current_messages = messages.copy()

        for step in range(5):  # Max 5 tool-call steps
            response = ollama.chat(
                model=OLLAMA_CHAT_MODEL,
                messages=current_messages,
                tools=TOOLS
            )
            msg = response["message"]

            # No tool calls → final answer
            if not msg.get("tool_calls"):
                return msg["content"] or "I couldn't find an answer."

            # Add assistant's tool-calling message to history
            current_messages.append({
                "role"      : "assistant",
                "content"   : msg.get("content", ""),
                "tool_calls": msg["tool_calls"]
            })

            # Execute each tool call
            for tc in msg["tool_calls"]:
                name   = tc["function"]["name"]
                args   = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)

                print(f"   🔧 Agent → {name}({args})")
                result = self._execute_tool(name, args)

                # Add tool result to messages
                current_messages.append({
                    "role"   : "tool",
                    "content": result
                })

        # Fallback: force final answer
        current_messages.append({
            "role"   : "user",
            "content": "Please provide your final answer based on all information gathered."
        })
        resp = ollama.chat(model=OLLAMA_CHAT_MODEL, messages=current_messages)
        return resp["message"]["content"]

    def _mistral_agent_loop(self, messages: list) -> str:
        """
        Mistral AI agentic loop using plain requests (no SDK needed).
        """
        from mistral_client import chat_complete
        current_messages = messages.copy()

        for step in range(5):
            response = chat_complete(
                model=MISTRAL_CHAT_MODEL,
                messages=current_messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            choice = response["choices"][0]
            msg = choice["message"]
            finish = choice["finish_reason"]

            # Final answer
            if finish == "stop" or not msg.get("tool_calls"):
                return msg.get("content") or "I could not find an answer."

            # Tool call step
            current_messages.append(msg)
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                print(f"   🔧 Agent → {name}({args})")
                result = self._execute_tool(name, args)
                current_messages.append({
                    "role"        : "tool",
                    "tool_call_id": tc["id"],
                    "content"     : result
                })

        return "I was unable to find a complete answer. Please try rephrasing your question."

    # ── PUBLIC CHAT INTERFACE ────────────────────────────────────

    def chat(self, user_message: str) -> tuple[str, list]:
        """
        Main entry point. Send a message, get back:
          - answer: str
          - tools_used: list of {"tool": ..., "args": ...}
        """
        # Clear tool log for this turn
        self.tool_call_log = []

        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_message})

        # Build full message list for the LLM
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + self.chat_history

        # Run agentic loop
        if LLM_BACKEND == "ollama":
            answer = self._ollama_agent_loop(messages)
        else:
            answer = self._mistral_agent_loop(messages)

        # Save assistant response to history
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer, self.tool_call_log.copy()

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []
        self.tool_call_log = []
        print("🔄 Chat history cleared.")

    def get_collection_stats(self) -> dict:
        """Return info about what's stored in ChromaDB."""
        if self.collection is None:
            return {"status": "empty", "count": 0}
        count = self.collection.count()
        return {
            "status"    : "ready",
            "count"     : count,
            "collection": CHROMA_COLLECTION,
            "path"      : CHROMA_PATH
        }

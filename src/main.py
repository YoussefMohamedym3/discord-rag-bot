import os
import sys

import phoenix as px

# 👇 CHANGE 1: Import the register function
from phoenix.otel import register

# (Keep your other imports like AppSettings, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import AppSettings
from src.indexing.bm25_indexer import build_bm25_index
from src.indexing.chroma_indexer import ingest_to_chroma
from src.preprocessing.parsing import run_cleaning_pipeline
from src.retrieval.retriever import HybridRAGRetriever
from src.services.rag_service import RAGService


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [clean|ingest|build-bm25|search|chat]")
        return

    # 👇 CHANGE 2: Use the register() pattern from your docs
    # This automatically connects LlamaIndex to the Phoenix UI
    tracer_provider = register(
        project_name="discord-rag-bot",  # We give it a specific name
        auto_instrument=True,  # This finds LlamaIndex automatically
    )

    # Launch the UI (keeping your persistent storage setting)
    session = px.launch_app(use_temp_dir=False)
    print(f"🚀 Phoenix Tracing running at: {session.url}")

    command = sys.argv[1]

    if command == "clean":
        run_cleaning_pipeline(AppSettings.DATA_RAW_DIR, AppSettings.DATA_SILVER_DIR)

    elif command == "ingest":
        ingest_to_chroma()
    elif command == "build-bm25":
        build_bm25_index()

    # --- NEW: CHAT COMMAND (Interactive Mode) ---
    elif command == "chat":
        print("🚀 Initializing Chatbot with Memory... (Type 'exit' to quit)")
        try:
            # Initialize the RAG Service (loads LLM + Retriever)
            bot_service = RAGService()
            print("✅ Bot Ready! Ask a question.")

            while True:
                user_input = input("\nUser (You): ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                # Get response from the service
                response = bot_service.chat_engine.chat(
                    user_input
                )  # Access engine directly to get objects
                print(f"🤖 Bot: {response}")

                # NEW: Print Sources used by the LLM
                if response.source_nodes:
                    print("\n📚 Sources Used:")
                    for node in response.source_nodes:
                        print(
                            f"   - {node.metadata.get('file_name')} (Score: {node.score:.2f})"
                        )

        except Exception as e:
            print(f"Error starting chat: {e}")
            print("Tip: Make sure your Docker 'vllm_server' is running!")

    # --- RETRIEVAL DEBUGGING (Keep this to test search quality separately) ---
    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide a query: python src/main.py search 'My Question'")
            return
        query = " ".join(sys.argv[2:])

        rag = HybridRAGRetriever()
        results = rag.retrieve(query)

        for i, node in enumerate(results, 1):
            source_file = node.metadata.get("file_name", "Unknown Source")
            print(f"\n--- Result {i} (Score: {node.score:.4f}) ---")
            print(f"📄 Source: {source_file}")
            print(node.text[:200] + "...")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

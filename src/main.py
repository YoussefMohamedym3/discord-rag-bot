import os
import sys

# Add the project root to python path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import AppSettings
from src.indexing.chroma_indexer import ingest_to_chroma
from src.preprocessing.parsing import run_cleaning_pipeline
from src.retrieval.retriever import RAGRetriever


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [clean|ingest|search <query>]")
        return

    command = sys.argv[1]

    if command == "clean":
        run_cleaning_pipeline(AppSettings.DATA_RAW_DIR, AppSettings.DATA_SILVER_DIR)

    elif command == "ingest":
        ingest_to_chroma()

    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide a query: python src/main.py search 'My Question'")
            return
        query = " ".join(sys.argv[2:])
        rag = RAGRetriever()
        results = rag.search(query)
        for i, node in enumerate(results, 1):
            # Extract filename from metadata
            source_file = node.metadata.get("file_name", "Unknown Source")

            print(f"\n--- Result {i} (Score: {node.score:.4f}) ---")
            print(f"📄 Source: {source_file}")  # <--- Add this line
            print(node.text[:200] + "...")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

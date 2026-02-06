import os
import pickle

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

from src.config.settings import AppSettings


def build_bm25_index():
    print("🏗️  Starting Node Extraction for BM25...")

    if not os.path.exists(AppSettings.DATA_SILVER_DIR):
        print(f"❌ Error: Data directory '{AppSettings.DATA_SILVER_DIR}' not found.")
        return

    # 1. Load and Parse Documents
    print(f"📖 Reading files from: {AppSettings.DATA_SILVER_DIR}")
    reader = SimpleDirectoryReader(
        input_dir=AppSettings.DATA_SILVER_DIR, recursive=True
    )
    documents = reader.load_data()

    parser = MarkdownNodeParser(include_metadata=True)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"🧩 Parsed {len(nodes)} nodes.")

    # 2. Save Nodes to Disk (Pickle)
    print(f"💾 Saving parsed nodes to: {AppSettings.NODES_INDEX_PATH}")
    os.makedirs(os.path.dirname(AppSettings.NODES_INDEX_PATH), exist_ok=True)

    with open(AppSettings.NODES_INDEX_PATH, "wb") as f:
        pickle.dump(nodes, f)

    print("✅ Nodes successfully saved! (Retriever will rebuild instantly from these)")


if __name__ == "__main__":
    build_bm25_index()

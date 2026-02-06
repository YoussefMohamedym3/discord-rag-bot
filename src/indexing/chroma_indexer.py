import os
import pickle

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.settings import AppSettings, setup_global_settings


def ingest_to_chroma():
    # 1. Initialize Settings (Load Embed Model)
    setup_global_settings()

    print(
        f"⚡ Connecting to ChromaDB at {AppSettings.CHROMA_HOST}:{AppSettings.CHROMA_PORT}..."
    )

    # 2. Connect to Chroma
    remote_db = chromadb.HttpClient(
        host=AppSettings.CHROMA_HOST, port=AppSettings.CHROMA_PORT
    )

    # Delete the old collection if it exists to prevent duplicate ghost nodes
    print(f"🧹 Checking for existing collection: '{AppSettings.COLLECTION_NAME}'...")
    try:
        remote_db.delete_collection(AppSettings.COLLECTION_NAME)
        print("   🗑️  Deleted existing collection. Starting fresh.")
    except ValueError:
        # Chroma raises ValueError if collection doesn't exist, which is fine
        print("   ✨ Collection does not exist yet. Creating new one.")

    chroma_collection = remote_db.get_or_create_collection(AppSettings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load the Master Nodes (Synced IDs)
    print(f"💾 Loading nodes from: {AppSettings.NODES_INDEX_PATH}")
    if not os.path.exists(AppSettings.NODES_INDEX_PATH):
        print(f"❌ Error: Node file not found at {AppSettings.NODES_INDEX_PATH}")
        print("👉 Run 'python src/main.py build-bm25' FIRST to generate the nodes.")
        return

    with open(AppSettings.NODES_INDEX_PATH, "rb") as f:
        nodes = pickle.load(f)

    print(f"🧩 Loaded {len(nodes)} nodes from disk.")

    # 4. Create Index (Generates Embeddings + Uploads)
    print("🚀 Starting Ingestion to Vector DB (This re-generates embeddings)...")

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print("✅ Ingestion Complete! Vector and BM25 indices are now 100% synced.")


if __name__ == "__main__":
    ingest_to_chroma()

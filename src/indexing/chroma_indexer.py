import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.settings import AppSettings, setup_global_settings


def ingest_to_chroma():
    # 1. Initialize Settings (Load Model)
    setup_global_settings()

    print(
        f"⚡ Connecting to ChromaDB at {AppSettings.CHROMA_HOST}:{AppSettings.CHROMA_PORT}..."
    )

    # 2. Connect to Chroma (Docker or Local)
    remote_db = chromadb.HttpClient(
        host=AppSettings.CHROMA_HOST, port=AppSettings.CHROMA_PORT
    )

    chroma_collection = remote_db.get_or_create_collection(AppSettings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"📖 Reading Silver Data from: {AppSettings.DATA_SILVER_DIR}")

    # 3. Load Data
    reader = SimpleDirectoryReader(
        input_dir=AppSettings.DATA_SILVER_DIR, recursive=True
    )
    documents = reader.load_data()

    # 4. Create Index (Chunk -> Embed -> Upload)
    markdown_parser = MarkdownNodeParser(include_metadata=True)

    print("🚀 Starting Ingestion (This might take a moment)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[markdown_parser],
        show_progress=True,
    )
    print("✅ Ingestion Complete!")


if __name__ == "__main__":
    ingest_to_chroma()

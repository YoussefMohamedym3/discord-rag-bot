import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables (from .env files)
load_dotenv()


class AppSettings:
    # Paths
    DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_SILVER_DIR = os.getenv("DATA_SILVER_DIR", "data/silver")

    # ChromaDB Settings
    # Default to 'localhost' for running scripts on your laptop
    # When running inside Docker, this will change to 'chromadb' (via .env)
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "discord_rag_bot")

    # Model Settings
    EMBED_MODEL_NAME = "BAAI/bge-m3"
    DEVICE = "cpu"  # Change to 'cpu' if testing on a non-GPU machine

    HYBRID_VECTOR_WEIGHT = 5.0
    HYBRID_BM25_WEIGHT = 3.0
    HYBRID_RECENCY_WEIGHT = 0.2


def setup_global_settings():
    """Initializes the LlamaIndex global settings."""
    print(
        f"⚙️  Loading Embedding Model: {AppSettings.EMBED_MODEL_NAME} on {AppSettings.DEVICE}..."
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=AppSettings.EMBED_MODEL_NAME,
        device=AppSettings.DEVICE,
        trust_remote_code=True,
    )
    # We disable the LLM here because we are only doing Data Science (Indexing/Retrieval)
    Settings.llm = None

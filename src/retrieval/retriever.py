import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.settings import AppSettings, setup_global_settings


class RAGRetriever:
    def __init__(self):
        setup_global_settings()

        # Connect to DB
        self.client = chromadb.HttpClient(
            host=AppSettings.CHROMA_HOST, port=AppSettings.CHROMA_PORT
        )
        self.collection = self.client.get_or_create_collection(
            AppSettings.COLLECTION_NAME
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

        # Load Index (Lightweight, doesn't re-embed)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
        )

    def search(self, query: str, top_k: int = 3):
        """Returns the top_k most relevant text chunks."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return nodes


# Simple test function
if __name__ == "__main__":
    rag = RAGRetriever()
    results = rag.search("What is the deadline for the cohort?")
    for node in results:
        print(f"\nScore: {node.score:.2f}")
        print(f"Text: {node.text[:100]}...")

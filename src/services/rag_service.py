# src/services/rag_service.py
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from src.config.settings import AppSettings
from src.retrieval.retriever import HybridRAGRetriever


class RAGService:
    def __init__(self):
        # 1. Initialize your Hybrid Retriever
        self.retriever = HybridRAGRetriever(top_k=3)

        # 2. Get the LLM (Llama-3.1 from Docker)
        self.llm = AppSettings.get_llm()

        # 3. Initialize Memory (Holds last ~5 turns)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # 4. Create the Chat Engine
        # This engine will:
        #   a. Take your new question + history
        #   b. Rewrite it into a standalone search query
        #   c. Use your Hybrid Retriever to find chunks
        #   d. Send chunks + query to LLM for final answer
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            memory=self.memory,
            system_prompt=(
                "You are a helpful AI Assistant for the PMA Bootcamp. "
                "Use the provided context to answer questions. "
                "CRITICAL: DO NOT SUMMARIZE LISTS OF LINKS. "
                "If the context provides a list of resources, videos, or tools, "
                "you must output EVERY SINGLE URL found in the context. "
                "Do not group them. List them individually with their full clickable markdown syntax. "
                "If you don't know the answer, say so."
            ),
            verbose=True,
        )

    # def chat(self, user_query: str) -> str:
    #     """
    #     Processes a user query with history and returns the response.
    #     """
    #     response = self.chat_engine.chat(user_query)
    #     return str(response)

    # In src/services/rag_service.py

    def chat(self, user_query: str):
        """
        Processes a user query with history and returns the FULL Response object.
        Do NOT wrap this in str(), or you lose the source nodes!
        """
        return self.chat_engine.chat(user_query)

    def reset_history(self):
        self.memory.reset()

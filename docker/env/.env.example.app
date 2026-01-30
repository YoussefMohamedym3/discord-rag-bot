# RAG Configuration
# Note: Using container names (chromadb, vllm) as hostnames within the network
CHROMA_HOST=chromadb
CHROMA_PORT=8000
COLLECTION_NAME=discord_rag_bot

# LLM Configuration
LLM_API_BASE=http://vllm:8000/v1
LLM_MODEL_NAME=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
# HF_TOKEN=your_huggingface_token_here
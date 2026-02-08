import json
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.services.rag_service import RAGService

router = APIRouter()

# --- 1. SESSION MANAGER ---
# This dictionary will hold one RAGService instance per user/session.
# In production, you'd use Redis or a database, but this works for a single Docker container.
sessions: Dict[str, RAGService] = {}


def get_service_for_session(session_id: str) -> RAGService:
    """Gets an existing service for a user or creates a new one."""
    if session_id not in sessions:
        print(f"✨ Creating new RAG session for ID: {session_id}")
        try:
            # Initialize a fresh service (with fresh memory) for this user
            sessions[session_id] = RAGService()
        except Exception as e:
            print(f"❌ Error creating service: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to initialize RAG service"
            )
    return sessions[session_id]


# --- 2. DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"  # Default allows testing without an ID


# --- 3. STANDARD ENDPOINT (Waits for full answer) ---
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    service = get_service_for_session(request.session_id)
    response = service.chat(request.query)

    # Extract sources
    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append(
                {
                    "file_name": node.metadata.get("file_name", "Unknown"),
                    "score": node.score if node.score else 0.0,
                    "text": node.text[:100] + "...",
                }
            )

    return {"answer": str(response), "sources": sources}


# --- 4. STREAMING ENDPOINT (Real-time) ---
@router.post("/chat/stream")
async def stream_chat_endpoint(request: ChatRequest):
    service = get_service_for_session(request.session_id)

    # We create a generator function that yields data chunk by chunk
    def iter_response():
        # Call the existing stream_chat method from your service
        streaming_response = service.stream_chat(request.query)

        # 1. Stream the text tokens
        for token in streaming_response.response_gen:
            # Yielding raw text (simple)
            yield token

        # 2. (Optional) Stream sources at the end?
        # Streaming is tricky; usually, we just stream text.
        # If you need sources, you might send a special separator like "\n---SOURCES---\n"
        if streaming_response.source_nodes:
            sources_json = json.dumps(
                [
                    n.metadata.get("file_name", "Unknown")
                    for n in streaming_response.source_nodes
                ]
            )
            yield f"\n\n[SOURCES: {sources_json}]"

    # Return the StreamingResponse
    return StreamingResponse(iter_response(), media_type="text/plain")

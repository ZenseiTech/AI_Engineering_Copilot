# app/agent/tools.py
from typing import Dict, Any, List, Callable
from app.core.database import AsyncSessionLocal
from app.rag.embeddings import get_gemini_embedding
import httpx
from app.rag.retriever import hybrid_retrieve_docs
import logging

logger = logging.getLogger(__name__)


# --- Tool Definitions ---
async def search_internal_docs(query: str) -> list[dict]:
    """
    Searches internal developer documentation via hybrid pgvector retrieval.
    """
    logger.info(f"Searching internal docs for query: {query}")
    # 1. Generate the query vector embedding
    query_vector = await get_gemini_embedding(query)

    # 2. Acquire an async DB session and call retriever with matching arguments
    async with AsyncSessionLocal() as db:
        results = await hybrid_retrieve_docs(
            db=db, query_text=query, query_vector=query_vector, top_k=5
        )

    return results


# --- Registry & Export ---

REGISTERED_TOOLS: List[Callable] = [
    search_internal_docs,
]

TOOL_MAP: Dict[str, Callable] = {func.__name__: func for func in REGISTERED_TOOLS}

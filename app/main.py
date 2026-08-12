import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.v1.chat import router as chat_router  # <--- 1. Import your router
from app.core.redis import redis_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to Redis
    await redis_manager.connect()
    yield
    # Shutdown: Clean up connections
    await redis_manager.disconnect()


app = FastAPI(title="Internal Dev Doc Copilot", lifespan=lifespan)

# <--- 2. Register your router with the matching prefix
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])


def hash_prompt(prompt: str) -> str:
    """Helper to generate a deterministic cache key from normalized prompt."""
    normalized = prompt.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.v1.chat import router as chat_router  # <--- 1. Import your router
from app.core.config import init_logging
from app.core.redis import redis_manager

import logging
from app.core.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize logging dictConfig from config.yaml on startup
    init_logging()

    logger = logging.getLogger(__name__)
    logger.info("Application starting up: %s v%s", config.app.name, config.app.version)

    # Startup: Connect to Redis
    await redis_manager.connect()
    yield
    # Shutdown: Clean up connections
    await redis_manager.disconnect()

    logger.info("Application shutting down")


app = FastAPI(title="Internal Dev Doc Copilot", lifespan=lifespan)

# <--- 2. Register your router with the matching prefix
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])


def hash_prompt(prompt: str) -> str:
    """Helper to generate a deterministic cache key from normalized prompt."""
    normalized = prompt.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

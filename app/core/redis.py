import json
import logging
from typing import Optional
import redis.asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger("copilot.redis")

# Default Redis URL (Docker container port)
REDIS_URL = "redis://localhost:6379/0"

class RedisManager:
    """
    Async Redis Manager for connection pooling, semantic query caching, 
    and fast session management.
    """
    def __init__(self):
        self.redis_client: Optional[Redis] = None

    async def connect(self, url: str = REDIS_URL):
        """Initialize the async Redis connection pool."""
        try:
            self.redis_client = aioredis.from_url(
                url, 
                encoding="utf-8", 
                decode_responses=True,
                max_connections=20
            )
            # Test ping
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    async def disconnect(self):
        """Gracefully close Redis connections on application shutdown."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed.")

    async def get_cached_response(self, query_hash: str) -> Optional[dict]:
        """
        Retrieves a cached answer for a developer query if available.
        """
        if not self.redis_client:
            return None
        try:
            data = await self.redis_client.get(f"doc_cache:{query_hash}")
            if data:
                logger.info(f"Cache HIT for query hash: {query_hash}")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis GET failed: {e}")
        return None

    async def set_cached_response(
        self, 
        query_hash: str, 
        response_payload: dict, 
        ttl_seconds: int = 86400  # Default 24-hour cache TTL
    ):
        """
        Caches a response payload to avoid re-running embeddings + LLM generation.
        """
        if not self.redis_client:
            return
        try:
            await self.redis_client.setex(
                f"doc_cache:{query_hash}",
                ttl_seconds,
                json.dumps(response_payload)
            )
            logger.info(f"Cached response for query hash: {query_hash}")
        except Exception as e:
            logger.warning(f"Redis SET failed: {e}")

# Global Redis Singleton instance
redis_manager = RedisManager()
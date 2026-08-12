import os
import yaml
import logging.config
from pathlib import Path
from pydantic import BaseModel, Field

# Locate project root directory (app/core/config.py -> parent x3 -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


class AppSettings(BaseModel):
    name: str
    version: str
    env: str
    debug: bool
    host: str
    port: int


class DatabaseSettings(BaseModel):
    host: str
    port: int
    user: str
    password: str
    name: str
    driver: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    echo: bool


class RedisSettings(BaseModel):
    host: str
    port: int
    db: int
    password: str
    ssl: bool
    max_connections: int
    socket_timeout: float
    default_ttl: int


class GeminiSettings(BaseModel):
    model: str
    embedding_model: str
    embedding_dimension: int
    temperature: float
    max_output_tokens: int


class RagSettings(BaseModel):
    top_k: int
    similarity_threshold: float


class GuardrailSettings(BaseModel):
    max_prompt_length: int
    strip_control_characters: bool


class Settings(BaseModel):
    app: AppSettings
    logging: dict
    database: DatabaseSettings
    redis: RedisSettings
    gemini: GeminiSettings
    rag: RagSettings
    guardrails: GuardrailSettings


def load_settings() -> Settings:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)
        raw_dict = yaml.safe_load(expanded_yaml)

    return Settings(**raw_dict)


# Global typed settings instance
config = load_settings()


def init_logging():
    """Initializes standard Python dictConfig logging from config.yaml"""
    os.makedirs(BASE_DIR / "logs", exist_ok=True)
    logging.config.dictConfig(config.logging)

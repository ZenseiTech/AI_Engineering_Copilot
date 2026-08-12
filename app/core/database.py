import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, Table, Column, Integer, String, Index, text
from pgvector.sqlalchemy import Vector

# Async Engine URL for Postgres with pgvector
DATABASE_URL = "postgresql+asyncpg://ai_user:ai_password@localhost:5431/copilot_db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_title: Mapped[str] = mapped_column(
        String(255), index=True
    )  # e.g. "payment-service-api.md"
    content: Mapped[str] = mapped_column(Text)
    repo_url: Mapped[str] = mapped_column(String(512), nullable=True)

    # 1536 dimensions for text-embedding-3-small or custom open models
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))


# Helper to initialize database extensions and HNSW index
async def init_db():
    async with engine.begin() as conn:
        # Chapter 3: Enable pgvector and full-text search extensions
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        await conn.run_sync(Base.metadata.create_all)

        # Add generated TSVector column for BM25 Keyword Search
        await conn.execute(text("""
            ALTER TABLE doc_chunks 
            ADD COLUMN IF NOT EXISTS content_tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
        """))

        # Create HNSW Vector Index for sub-10ms cosine similarity (Chapter 10)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_hnsw_embedding 
            ON doc_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))

        # Create GIN Index for BM25 Full-Text Search
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_gin_content_tsv 
            ON doc_chunks USING gin (content_tsv);
        """))

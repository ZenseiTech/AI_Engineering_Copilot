import os
import asyncio
import logging
from typing import List, Dict, Any
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from google import genai
from google.genai import types

from app.core.database import AsyncSessionLocal, DocumentChunk

logger = logging.getLogger("copilot.github_sync")
client = genai.Client()

# Configuration
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN", ""
)  # Optional for public repos, recommended for rate limits
TARGET_REPO = os.getenv("GITHUB_REPO", "octocat/Hello-World")  # owner/repo format
BRANCH = os.getenv("GITHUB_BRANCH", "main")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# Helper: Recursive Markdown Chunking with Overlap
def markdown_overlap_chunker(
    text_content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Splits markdown documents into overlapping chunks while preferring natural boundary splits
    such as headers (##), paragraphs (\n\n), and lines (\n).
    """
    if not text_content or not text_content.strip():
        return []

    # Standard Markdown separators ordered by structural priority
    separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]

    def split_text(text: str, seps: List[str]) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        if not seps:
            # Fallback character split
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)
            ]

        sep = seps[0]
        splits = text.split(sep)
        chunks = []
        current_chunk = ""

        for i, piece in enumerate(splits):
            candidate = (current_chunk + sep + piece) if current_chunk else piece
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Apply sliding window overlap from the end of the previous chunk
                overlap_text = (
                    current_chunk[-overlap:]
                    if len(current_chunk) > overlap
                    else current_chunk
                )
                current_chunk = (overlap_text + sep + piece) if overlap_text else piece

        if current_chunk:
            chunks.append(current_chunk)

        # Refine any chunks that still exceed maximum size using lower priority separators
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                final_chunks.extend(split_text(chunk, seps[1:]))
            else:
                final_chunks.append(chunk)

        return final_chunks

    return split_text(text_content, separators)


# GitHub API Document Fetcher
async def fetch_github_markdown_files(
    repo: str, branch: str = "main"
) -> List[Dict[str, str]]:
    """
    Recursively fetches all .md and .mdx files from a GitHub repository tree.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as http_client:
        response = await http_client.get(tree_url)
        if response.status_code != 200:
            logger.error(
                f"Failed to fetch GitHub repository tree: {response.status_code} {response.text}"
            )
            return []

        tree_data = response.json()
        md_files = [
            item
            for item in tree_data.get("tree", [])
            if item["type"] == "blob"
            and (item["path"].endswith(".md") or item["path"].endswith(".mdx"))
        ]

        documents = []
        for file_info in md_files:
            file_path = file_info["path"]
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"

            raw_resp = await http_client.get(raw_url)
            if raw_resp.status_code == 200:
                documents.append(
                    {
                        "title": file_path.split("/")[-1],
                        "path": file_path,
                        "repo_url": f"https://github.com/{repo}/blob/{branch}/{file_path}",
                        "content": raw_resp.text,
                    }
                )
                logger.info(f"Fetched document: {file_path}")
            else:
                logger.warning(f"Could not fetch content for: {file_path}")

        return documents


# Main Sync Orchestrator
async def sync_github_to_pgvector(repo: str = TARGET_REPO, branch: str = BRANCH):
    """
    Pulls docs from GitHub, chunks with overlap, generates Gemini embeddings, and syncs pgvector.
    """
    logger.info(f"Starting GitHub sync for {repo} ({branch})...")
    docs = await fetch_github_markdown_files(repo, branch)

    if not docs:
        logger.warning("No Markdown files found to process.")
        return

    # Extract target identifiers safely from the model definition
    table_name = DocumentChunk.__tablename__
    schema_prefix = (
        f"{DocumentChunk.__table__.schema}." if DocumentChunk.__table__.schema else ""
    )
    full_table_path = f"{schema_prefix}{table_name}"

    async with AsyncSessionLocal() as db:
        # Step 1: Collect active document paths to detect and purge deleted files in repo
        active_repo_urls = [d["repo_url"] for d in docs]

        # Purge stale database chunks no longer in GitHub repo
        delete_stale_sql = text(
            f"DELETE FROM {full_table_path} WHERE NOT (repo_url = ANY(:active_urls))"
        )
        await db.execute(delete_stale_sql, {"active_urls": list(active_repo_urls)})
        await db.commit()

        # Step 2: Chunk and Embed updated documents
        for doc in docs:
            logger.info(f"Processing & Chunking: {doc['path']}")

            # Delete existing chunks for this specific document before re-inserting (Idempotent update)
            await db.execute(
                text(f"DELETE FROM {full_table_path} WHERE repo_url = :repo_url"),
                {"repo_url": doc["repo_url"]},
            )

            # Generate overlapping chunks
            chunks = markdown_overlap_chunker(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Fetch embeddings from stable Gemini model
                embed_response = await client.aio.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk_text,
                    config={"output_dimensionality": 1536},
                )

                # FIXED: Extracted `.values` from the first element of the embeddings list object
                vector_embedding = embed_response.embeddings[0].values

                # Save Chunk to Postgres
                new_chunk = DocumentChunk(
                    doc_title=f"{doc['title']} (Part {chunk_idx + 1})",
                    content=chunk_text,
                    repo_url=doc["repo_url"],
                    embedding=vector_embedding,
                )
                db.add(new_chunk)

            await db.commit()
            logger.info(f"Successfully synced {len(chunks)} chunks for {doc['title']}")

    logger.info("GitHub ingestion sync completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_github_to_pgvector())

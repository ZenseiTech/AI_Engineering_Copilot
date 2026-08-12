import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def hybrid_retrieve_docs(
    db: AsyncSession, query_text: str, query_vector: list[float], top_k: int = 5
) -> list[dict]:
    """
    Performs Hybrid Search (Dense HNSW Vector + Sparse BM25) with RRF scoring inside Postgres.
    """
    sql_query = text("""
        WITH vector_hits AS (
            SELECT id, doc_title, content, repo_url,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> :vector) AS rank
            FROM doc_chunks
            ORDER BY embedding <=> :vector
            LIMIT 20
        ),
        keyword_hits AS (
            SELECT id, doc_title, content, repo_url,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', :query)) DESC) AS rank
            FROM doc_chunks
            WHERE content_tsv @@ plainto_tsquery('english', :query)
            LIMIT 20
        )
        SELECT 
            COALESCE(v.id, k.id) AS id,
            COALESCE(v.doc_title, k.doc_title) AS doc_title,
            COALESCE(v.content, k.content) AS content,
            COALESCE(v.repo_url, k.repo_url) AS repo_url,
            (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score
        FROM vector_hits v
        FULL OUTER JOIN keyword_hits k ON v.id = k.id
        ORDER BY rrf_score DESC
        LIMIT :top_k;
    """)

    # Format vector as postgres array string
    vector_str = f"[{','.join(map(str, query_vector))}]"

    # Sanitize query string for fulltext
    sanitized_query = " & ".join(re.findall(r"\w+", query_text))
    if not sanitized_query:
        sanitized_query = query_text

    result = await db.execute(
        sql_query, {"vector": vector_str, "query": sanitized_query, "top_k": top_k}
    )

    rows = result.fetchall()
    return [
        {
            "id": row.id,
            "doc_title": row.doc_title,
            "content": row.content,
            "repo_url": row.repo_url,
            "score": float(row.rrf_score),
        }
        for row in rows
    ]

import json
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from google import genai
from google.genai import types

from app.core.database import AsyncSessionLocal
from app.guardrails.input_filter import sanitize_dev_prompt
from app.rag.retriever import hybrid_retrieve_docs
from app.rag.embeddings import get_gemini_embedding

router = APIRouter()

# Initialize Gemini Client
client = genai.Client()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


@router.get("/chat/stream")
async def stream_dev_assistant(
    prompt: str = Query(..., description="Developer question"),
    db: AsyncSession = Depends(get_db),
):
    # 1. Apply Input Guardrails & PII Redaction
    clean_prompt = sanitize_dev_prompt(prompt)

    # 2. Generate Real Gemini Embedding & Retrieve Context via Hybrid pgvector
    try:
        query_vector = await get_gemini_embedding(clean_prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate embedding: {str(e)}"
        )

    retrieved_chunks = await hybrid_retrieve_docs(
        db, clean_prompt, query_vector, top_k=3
    )

    # 3. Format Documentation Context Block
    context_str = (
        "\n\n".join(
            [
                f"--- Source: {c['doc_title']} ({c['repo_url']}) ---\n{c['content']}"
                for c in retrieved_chunks
            ]
        )
        if retrieved_chunks
        else "No relevant internal documentation found."
    )

    system_instruction = f"""You are an expert Internal Developer Documentation Copilot.
Answer the developer's question accurately using ONLY the provided documentation context below.
If the context doesn't contain the answer, state that clearly and point them to the relevant engineering team or repository.

### Context:
{context_str}
"""

    # 4. Stream Tokens directly from Gemini 2.5 Flash via Open SDK
    async def token_generator():
        try:
            # Async streaming call to Gemini API
            response_stream = await client.aio.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=clean_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,  # Low temperature for accurate technical docs
                    max_output_tokens=1024,
                ),
            )

            async for chunk in response_stream:
                if chunk.text:
                    # Yield token formatted as SSE JSON payload
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"

        except Exception as err:
            yield f"data: {json.dumps({'error': f'Gemini Stream Error: {str(err)}'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

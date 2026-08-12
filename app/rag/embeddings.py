from google import genai
from google.genai import types

# Initialize the Gemini Client (reads GEMINI_API_KEY from environment)
client = genai.Client()


async def get_gemini_embedding(text: str) -> list[float]:
    """
    Generates a 1536-dimensional embedding using Gemini's text-embedding-001 model asynchronously.
    """
    response = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            # Matching standard dimension size for DB schema compatibility
            output_dimensionality=1536
        ),
    )

    # Extract raw float vector
    return response.embeddings[0].values

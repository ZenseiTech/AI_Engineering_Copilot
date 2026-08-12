# Enterprise Agent System

### System Architecture

    ┌────────────────────────────────────────────────────────────────────────┐
    │                        CLIENT (Web / Desktop UI)                       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ SSE Streaming Request
    ┌───────────────────────────────────▼────────────────────────────────────┐
    │  1. API & GUARDRAIL LAYER (FastAPI Async)                             │
    │     • Input Safety Interceptor & PII Redaction                         │
    │     • Redis Semantic Cache Lookup (Sub-10ms response if hit)           │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Cache Miss
    ┌───────────────────────────────────▼────────────────────────────────────┐
    │  2. RECOVERY & ROUTING GATEWAY                                         │
    │     • Primary Model (e.g., vLLM / OpenAI) ──> Fallback Provider Routing  │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            │                                                       │
    ┌───────▼───────────────────────────────┐       ┌───────────────▼────────────────────────┐
    │ 3. HYBRID RAG RETRIEVER (pgvector)    │       │ 4. NATIVE TOOL / AGENT LOOP            │
    │    • Dense Vector + BM25 Search       │       │    • Live API Web Search / Calculator  │
    │    • Cross-Encoder Reranking          │       │    • System Execution Scratchpad       │
    └───────────────────────────────────────┘       └────────────────────────────────────────┘

### Directory Tree

    ai-engineering-copilot/
    ├── docker-compose.yml      # Postgres + pgvector, Redis
    ├── Dockerfile              # Multi-stage production container
    ├── requirements.txt        # FastAPI, pgvector, httpx, pydantic
    ├── .env.example            # API keys and DB credentials
    └── app/
    ├── main.py                 # FastAPI app entry point & lifecycle
    ├── core/
    │ ├── config.py             # Pydantic environment settings
    │ ├── database.py           # Async SQLAlchemy 2.0 + pgvector setup
    │ └── redis.py              # Async Redis cache connection
    ├── guardrails/
    │ ├── input_filter.py       # Prompt injection & PII regex check
    │ └── output_filter.py      # Schema validation & hallucination checks
    ├── rag/
    │ ├── embeddings.py         # Async embedding generation
    │ └── retriever.py          # pgvector hybrid search & reranker
    ├── agents/
    │ ├── tools.py              # Executable function definitions
    │ └── loop.py               # Async ReAct loop & LLM streaming
    └── api/
        └── v1/
            └── chat.py         # Streaming POST /chat endpoint

### Testing our service

1.  Start Infrastructure:

        docker compose up -d

2.  Export env variables

        export GITHUB_TOKEN="your_token"
        export GITHUB_REPO="your-org/your-docs-repo"
        export GITHUB_BRANCH="master"
        export GEMINI_API_KEY="your-gemini-api-key"

3.  Initialiaze Schema and Indexes

        import asyncio
        from app.core.database import init_db

        asyncio.run(init_db())

4.  To run the service

        fastapi dev main.py

        If you are using Uvicorn directly, run:

            uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

5.  Query stream endpoint via Curl

        curl -N "http://localhost:8000/api/v1/chat/stream?prompt=How%20to%20build%020project?"

### Docker info

$ docker info | grep "Docker Root Dir"

    Docker Root Dir: /media/zensei/MY_PASSPORT/external-docker/docker-data

### Executing Ingestion Script

    python -m app.ingestion.github_sync

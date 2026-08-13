# app/api/v1/chat.py
from app.guardrails.input_filter import sanitize_dev_prompt
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.agent.loop import run_agent_loop
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_INSTRUCTION = "You are an internal Developer Copilot equipped with staging API and doc search tools."


@router.get("/chat/agent")
async def chat_agent(prompt: str = Query(...)):

    logger.info(f"Received prompt: {prompt}")

    # 1. Apply Input Guardrails & PII Redaction
    clean_prompt = sanitize_dev_prompt(prompt)

    return StreamingResponse(
        run_agent_loop(prompt=clean_prompt, system_instruction=SYSTEM_INSTRUCTION),
        media_type="text/event-stream",
    )

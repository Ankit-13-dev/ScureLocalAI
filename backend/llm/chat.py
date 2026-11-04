from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from backend.config import LlmSettings, get_settings


logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I could not find relevant information in the knowledge base."

RAG_SYSTEM_PROMPT = """You are a research assistant. Answer ONLY using the provided document excerpts.
If the excerpts do not contain enough information to answer the question, respond with:
"I could not find relevant information in the knowledge base."
Always end your answer with a "Sources:" section listing the filenames you used."""


SIMPLE_SYSTEM_PROMPT = """You are a friendly tutor helping a student understand their study material.

Rules you must follow:
1. Use simple everyday language - no jargon. If you must use a technical word, explain it immediately in plain words inside brackets.
2. Structure your answer as short bullet points or numbered steps - never long dense paragraphs.
3. Use a real-life analogy or example to make every abstract idea concrete.
4. Keep each sentence under 20 words.
5. Answer ONLY using the provided document excerpts - never invent information.
6. If the excerpts do not contain the answer, say: "I couldn't find this in your uploaded notes. Try asking your teacher or checking another chapter."
7. End every answer with: "Quick check: [write one simple question the student can ask themselves to confirm they understood]"

Always end with a "Sources:" section listing the filenames you used."""


def _pick_system_prompt(mode: str) -> str:
    """Normal = precise research assistant; "simple" = plain-language tutor."""
    return SIMPLE_SYSTEM_PROMPT if str(mode).lower() == "simple" else RAG_SYSTEM_PROMPT


def _format_context(context_chunks: list[dict[str, Any]]) -> str:
    excerpts: list[str] = []
    for index, chunk in enumerate(context_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")
        excerpts.append(f"[{index}] Source: {filename}, page {page}\n{text}")
    return "\n\n".join(excerpts)


def _source_names(context_chunks: list[dict[str, Any]]) -> list[str]:
    names = []
    for chunk in context_chunks:
        filename = str(chunk.get("filename", "")).strip()
        if filename and filename not in names:
            names.append(filename)
    return names


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def ensure_sources(answer: str, context_chunks: list[dict[str, Any]]) -> str:
    sources = _source_names(context_chunks)
    if not sources:
        return answer

    if "Sources:" in answer:
        return answer.strip()

    source_lines = "\n".join(f"- {source}" for source in sources)
    return f"{answer.strip()}\n\nSources:\n{source_lines}"


async def answer_query(
    question: str,
    context_chunks: list[dict[str, Any]],
    settings: LlmSettings | None = None,
    mode: str = "normal",
) -> str:
    if not context_chunks:
        return NOT_FOUND_MESSAGE

    resolved = settings or get_settings().llm
    context = _format_context(context_chunks)
    messages = [
        {"role": "system", "content": f"{_pick_system_prompt(mode)}\n\nDocument excerpts:\n{context}"},
        {"role": "user", "content": question},
    ]
    payload = {
        "model": resolved.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": resolved.temperature,
            "num_predict": resolved.max_tokens,
        },
    }

    # 360 s: a larger local model on a modest GPU/CPU with many retrieved chunks can take minutes
    async with httpx.AsyncClient(timeout=360.0) as client:
        response = await client.post(f"{resolved.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        answer = str(data["message"]["content"])
    except KeyError as exc:
        logger.error("Unexpected Ollama chat response: %s", data)
        raise RuntimeError("Ollama returned an unexpected chat response") from exc

    return ensure_sources(_strip_thinking(answer), context_chunks)

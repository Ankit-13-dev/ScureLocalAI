from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.llm.chat import answer_query
from backend.retrieval.embedder import embed
from backend.retrieval.figure_search import (
    IMAGE_NOT_FOUND_MESSAGE,
    NO_IMAGES_INDEXED_MESSAGE,
    analyze_question,
    build_image_reply,
    figure_context_chunks,
    find_figures,
    to_image_ref,
)
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)


class ImageRef(BaseModel):
    id: str
    doc_id: str
    figure_id: str
    filename: str
    page: int | None = None
    kind: str = "unknown"
    caption: str | None = None
    url: str
    match: str = "semantic"
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    images: list[ImageRef] = Field(default_factory=list)


def _sources(chunks: list[dict[str, object]]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        filename = str(chunk.get("filename", "")).strip()
        if filename and filename not in sources:
            sources.append(filename)
    return sources


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    try:
        question_embedding = (await embed([question], settings.embeddings))[0]

        intent = analyze_question(question)
        figures: list[dict[str, object]] = []
        if settings.figures.enabled:
            try:
                figures = find_figures(question, intent, question_embedding, store, settings.figures)
            except Exception:  # noqa: BLE001 - a picture-search problem must never break text answers
                logger.exception("Image lookup failed; continuing with a text-only answer")

        if intent.wants_images and not intent.wants_text:
            # "Show me X": the picture is the answer. No LLM call, so nothing can be invented.
            if figures:
                answer = build_image_reply(figures)
            elif store.figure_counts():
                answer = IMAGE_NOT_FOUND_MESSAGE
            else:
                answer = NO_IMAGES_INDEXED_MESSAGE
            sources = _sources(figures)
        else:
            chunks = store.query(question_embedding, request.k or settings.retrieval.top_k)
            context = chunks + figure_context_chunks(figures)
            answer = await answer_query(question, context, settings.llm)
            if intent.wants_images and not figures:
                answer = f"{answer}\n\n({IMAGE_NOT_FOUND_MESSAGE})"
            sources = _sources(context)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to answer query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer query: {exc}",
        ) from exc

    return QueryResponse(answer=answer, sources=sources, images=[to_image_ref(f) for f in figures])

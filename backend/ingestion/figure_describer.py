from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from backend.config import VisionSettings, get_settings

logger = logging.getLogger(__name__)

FIGURE_DESCRIPTION_PROMPT = """Describe this PDF figure for a document question-answering system.
State the main subject, important visible components, labels, relationships, axes/units or trends when present, and what the figure communicates. Use only what is visible; do not guess unreadable labels. Be concise (2-5 sentences). Do not mention that you are an AI. Return only the description."""


async def describe_figure(
    image_path: Path,
    caption: str = "",
    kind: str = "unknown",
    settings: VisionSettings | None = None,
) -> str:
    """Generate a grounded description of one image using a local Ollama VLM."""
    resolved = settings or get_settings().vision
    if resolved is None or not resolved.enabled:
        return ""
    if not image_path.is_file():
        raise FileNotFoundError(f"Figure crop not found: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    context = f"Detected figure type: {kind}."
    if caption.strip():
        context += f" PDF caption near the figure: {caption.strip()}"
    payload = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": FIGURE_DESCRIPTION_PROMPT},
            {"role": "user", "content": context, "images": [encoded]},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": resolved.max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.post(f"{resolved.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not describe PDF figures with Ollama model '{resolved.model}'. "
            f"Ensure Ollama is running and run `ollama pull {resolved.model}`. Details: {exc}"
        ) from exc

    description = str(data.get("message", {}).get("content", "")).strip()
    if not description:
        raise RuntimeError(f"Ollama vision model '{resolved.model}' returned an empty figure description")
    return description


async def describe_figures(
    figure_records: list[dict[str, Any]],
    figure_dir: Path,
    settings: VisionSettings | None = None,
) -> list[dict[str, Any]]:
    """Describe figure records sequentially to keep local VRAM usage predictable."""
    resolved = settings or get_settings().vision
    if resolved is None or not resolved.enabled:
        return figure_records

    for record in figure_records:
        relative_crop = str(record.get("crop_path", ""))
        image_path = (figure_dir / relative_crop).resolve()
        # Enforce that manifest paths cannot escape this document's figure folder.
        if figure_dir.resolve() not in image_path.parents:
            raise ValueError(f"Invalid figure crop path: {relative_crop}")
        record["description"] = await describe_figure(
            image_path,
            caption=str(record.get("caption") or ""),
            kind=str(record.get("kind") or "unknown"),
            settings=resolved,
        )
        logger.info("Described figure %s on page %s", record.get("figure_id"), record.get("page"))
    return figure_records

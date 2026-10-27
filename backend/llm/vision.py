"""Optional local vision-language model (VLM) that writes a short description of a picture.

The description is stored next to the picture so that questions such as "show me the picture
with the bird" can find images that have no printed caption.  Everything runs through the same
local Ollama server as the chat model - nothing leaves the machine.

This module is deliberately failure-tolerant: if the vision model is not installed or Ollama is
unreachable, descriptions are skipped (a single warning is logged) and image retrieval keeps
working from captions and surrounding text.
"""

from __future__ import annotations

import base64
import logging
import re

import httpx

from backend.config import Settings


logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "You are indexing a picture cut out of a document so it can be found by search. "
    "In 2-3 plain sentences say what kind of visual it is (photograph, bar chart, line graph, "
    "pie chart, flowchart, diagram, screenshot, map, ...), what it shows, and copy any visible "
    "title, axis names, legend entries or labels. Describe only what is visible; do not guess."
)


class VisionDescriber:
    """Calls an Ollama vision model. ``describe`` returns None when it cannot help."""

    def __init__(self, settings: Settings) -> None:
        self.mode = settings.figures.describe_mode
        self.model = settings.figures.vlm_model
        self.base_url = (settings.figures.vlm_base_url or settings.llm.base_url).rstrip("/")
        self.timeout = settings.figures.vlm_timeout_s
        self._broken = False

    @property
    def active(self) -> bool:
        return self.mode in ("uncaptioned", "all") and not self._broken

    def wants(self, caption: str | None) -> bool:
        if not self.active:
            return False
        return self.mode == "all" or not (caption or "").strip()

    async def describe(self, image_png: bytes) -> str | None:
        if not self.active:
            return None
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": DESCRIBE_PROMPT,
                    "images": [base64.b64encode(image_png).decode("ascii")],
                }
            ],
            "options": {"temperature": 0.1, "num_predict": 220},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                content = str(response.json()["message"]["content"])
        except Exception as exc:  # noqa: BLE001 - descriptions are optional, never fatal
            self._broken = True
            logger.warning(
                "Vision model '%s' unavailable (%s). Image descriptions are skipped; run "
                "'ollama pull %s' to enable them.",
                self.model,
                exc,
                self.model,
            )
            return None

        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = " ".join(text.split())
        return text[:700] or None

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

import cv2
import httpx
import numpy as np

from backend.config import Settings


logger = logging.getLogger(__name__)

BRIEF_PROMPT = (
    "You are indexing a picture cut out of a document so it can be found by search. "
    "In 2-3 plain sentences say what kind of visual it is (photograph, bar chart, line graph, "
    "pie chart, flowchart, diagram, screenshot, map, ...), what it shows, and copy any visible "
    "title, axis names, legend entries or labels. Describe only what is visible; do not guess."
)

# Exhaustive, sectioned description (people, clothing colours, counts, text, diagrams).  Written so that
# factual questions such as "how many children are shown?" or "what colour is the top?" can be
# answered from the description alone.
DETAILED_PROMPT = (
    "Describe this image in complete detail using these exact labeled sections:\n\n"
    "PEOPLE: For every person visible, state their role if identifiable (teacher/student/etc), "
    "gender, approximate age, EXACT clothing colours and type for each garment separately "
    "(e.g. 'wearing a GREEN top, BLUE jeans, and WHITE shoes'). "
    "State the EXACT COUNT of people visible. Write 'none' if there are no people.\n\n"
    "OBJECTS: Name every object, its colour, size, and position.\n\n"
    "TEXT: Transcribe every word of text visible in the image exactly.\n\n"
    "COLOURS: List the dominant colours. For each person state clothing colours explicitly.\n\n"
    "LAYOUT: Describe the composition and arrangement.\n\n"
    "DIAGRAMS: If this is a technical diagram or chart, describe every label, arrow, box, "
    "connection, axis, legend entry and value shown.\n\n"
    "Be exhaustive - every detail that could answer a factual question must be stated explicitly. "
    "Describe only what is visible; do not guess."
)


class VisionDescriber:
    """Calls an Ollama vision model. ``describe`` returns None when it cannot help."""

    def __init__(self, settings: Settings) -> None:
        self.mode = settings.figures.describe_mode
        self.model = settings.figures.vlm_model
        self.base_url = (settings.figures.vlm_base_url or settings.llm.base_url).rstrip("/")
        self.timeout = settings.figures.vlm_timeout_s
        self.detailed = settings.figures.describe_prompt != "brief"
        self.max_px = max(128, int(settings.figures.vlm_max_px))
        self._broken = False

    @property
    def active(self) -> bool:
        return self.mode in ("uncaptioned", "all") and not self._broken

    def wants(self, caption: str | None) -> bool:
        if not self.active:
            return False
        return self.mode == "all" or not (caption or "").strip()

    def _prepare(self, image_png: bytes) -> bytes:
        """Shrink big crops so the vision model runs faster (detail beyond ~768 px rarely helps)."""
        try:
            image = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return image_png
            height, width = image.shape[:2]
            if max(height, width) <= self.max_px:
                return image_png
            ratio = self.max_px / float(max(height, width))
            resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", resized)
            return buf.tobytes() if ok else image_png
        except Exception:  # noqa: BLE001 - fall back to the original bytes
            return image_png

    async def describe(self, image_png: bytes) -> str | None:
        if not self.active:
            return None
        image_png = self._prepare(image_png)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": DETAILED_PROMPT if self.detailed else BRIEF_PROMPT,
                    "images": [base64.b64encode(image_png).decode("ascii")],
                }
            ],
            "options": {"temperature": 0.1, "num_predict": 800 if self.detailed else 220},
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
        return text[: 3000 if self.detailed else 700] or None

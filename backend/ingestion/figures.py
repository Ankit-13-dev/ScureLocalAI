"""Turn the pictures found by Visualizing.py into searchable, displayable records.

Flow for every uploaded PDF:
    Visualizing.detect_figures  -> where are the pictures / charts / diagrams?
    save_crops                  -> one PNG per picture in data/figures/<doc_id>/crops/
    text for search             -> caption + sentences that cite "Fig. N" + nearby text
                                   (+ optional local vision-model description)
    embed + ChromaStore         -> separate "<collection>_figures" collection

At question time (see backend/retrieval/figure_search.py) the best-matching records are
returned and the chat UI shows the stored crops.  Nothing here generates images.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.llm.vision import VisionDescriber
from backend.retrieval.embedder import embed
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)

FIG_NUMBER_RE = re.compile(r"\bfig(?:ure)?s?\.?\s*#?(\d{1,3})", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

KIND_WORDS = {
    "photo": "photograph picture photo image",
    "chart": "chart graph plot figure",
    "diagram_or_illustration": "diagram illustration drawing figure flowchart",
}


def caption_number(caption: str | None) -> int | None:
    match = FIG_NUMBER_RE.search(caption or "")
    return int(match.group(1)) if match else None


def _mentions(page_texts: list[str], number: int | None, caption: str | None, limit: int = 2) -> str:
    """Body sentences that cite the same figure number ("as shown in Fig. 3, ...")."""
    if number is None:
        return ""
    head = (caption or "")[:30]
    found: list[str] = []
    for text in page_texts:
        for sentence in _SENTENCE_SPLIT.split(" ".join(text.split())):
            if head and head in sentence:  # the caption itself, not a citation of it
                continue
            match = FIG_NUMBER_RE.search(sentence)
            if match and int(match.group(1)) == number and 5 <= len(sentence.split()) <= 60:
                found.append(sentence)
                if len(found) >= limit:
                    return " ".join(found)
    return " ".join(found)


def _nearby_text(blocks: list[tuple], bbox: tuple, caption_bbox: tuple | None, limit: int = 300) -> str:
    """Short text directly above / below the picture (titles, legends, lead-in sentences)."""
    x0, y0, x1, y1 = bbox
    picked: list[tuple[float, str]] = []
    for b in blocks:
        if len(b) >= 7 and b[6] != 0:
            continue
        bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], " ".join(str(b[4]).split())
        if not text:
            continue
        if caption_bbox and abs(by0 - caption_bbox[1]) < 2 and abs(bx0 - caption_bbox[0]) < 2:
            continue
        overlap = min(x1, bx1) - max(x0, bx0)
        if overlap < 0.3 * min(x1 - x0, bx1 - bx0):
            continue
        gap = min(abs(by0 - y1), abs(y0 - by1))
        inside = by0 >= y0 and by1 <= y1
        if gap <= 40 and not inside and len(text.split()) <= 30:  # titles / legends, not body paragraphs
            picked.append((gap, text))
    picked.sort(key=lambda item: item[0])
    return " ".join(text for _gap, text in picked)[:limit]


def extract_figures(pdf_path: Path, doc_id: str, filename: str, settings: Settings) -> list[dict[str, Any]]:
    """Detect pictures, write crops and build the searchable text. Blocking (CPU) work."""
    import pymupdf

    from backend.ingestion.Visualizing import detect_figures, load_vision_config, save_crops

    out_dir = settings.figures_root / doc_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    cfg = load_vision_config()
    result = detect_figures(pdf_path, cfg)
    if not result.figures:
        return []
    save_crops(pdf_path, result, out_dir, cfg)

    doc = pymupdf.open(str(pdf_path))
    try:
        page_texts = [page.get_text("text") for page in doc]
        records: list[dict[str, Any]] = []
        for page_result in result.pages:
            blocks = doc[page_result.page - 1].get_text("blocks") if page_result.figures else []
            for fig in page_result.figures:
                if not fig.crop_path or not Path(fig.crop_path).exists():
                    continue
                number = caption_number(fig.caption)
                mentions = _mentions(page_texts, number, fig.caption)
                nearby = _nearby_text(blocks, fig.bbox, fig.caption_bbox)
                records.append(
                    {
                        "figure_id": fig.figure_id,
                        "page": fig.page,
                        "kind": fig.kind,
                        "caption": fig.caption or "",
                        "fig_number": number,
                        "bbox": fig.bbox,
                        "source": fig.source,
                        "confidence": fig.confidence,
                        "crop_file": f"{doc_id}/crops/{fig.figure_id}.png",
                        "crop_path": fig.crop_path,
                        "mentions": mentions,
                        "nearby": nearby,
                        "description": "",
                    }
                )
    finally:
        doc.close()
    return records


def build_search_text(record: dict[str, Any], filename: str) -> str:
    parts = [
        KIND_WORDS.get(str(record.get("kind")), "figure image picture"),
        str(record.get("caption") or ""),
        str(record.get("description") or ""),
        f"Referred to in the text: {record['mentions']}" if record.get("mentions") else "",
        f"Nearby text: {record['nearby']}" if record.get("nearby") else "",
        f"Page {record.get('page')} of {filename}",
    ]
    return ". ".join(part.strip() for part in parts if part and part.strip())


async def ingest_figures(
    file_path: Path,
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
) -> int:
    """(Re)index every picture of one PDF. Safe to call repeatedly. Returns the number indexed."""
    resolved = settings or get_settings()
    if not resolved.figures.enabled or file_path.suffix.lower() != ".pdf":
        return 0
    target = store or get_store()

    records = await asyncio.to_thread(extract_figures, file_path, doc_id, filename, resolved)
    target.delete_figures(doc_id)  # drop stale entries from an earlier scan
    if not records:
        return 0

    describer = VisionDescriber(resolved)
    for record in records:
        if describer.wants(record["caption"]):
            image = Path(record["crop_path"]).read_bytes()
            record["description"] = await describer.describe(image) or ""

    for record in records:
        record["search_text"] = build_search_text(record, filename)
    embeddings = await embed([record["search_text"] for record in records], resolved.embeddings)
    target.add_figures(doc_id, filename, records, embeddings)
    logger.info("Indexed %d image(s) from %s", len(records), filename)
    return len(records)

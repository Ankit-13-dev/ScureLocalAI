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
from backend.ingestion.structure import DocumentStructure, annotate_figures, options_from_settings, safe_extract_structure
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


def figure_records(pdf_path: Path, result: Any, doc_id: str, require_crop: bool = False) -> list[dict[str, Any]]:
    """Turn Visualizing's detections into plain records (caption number, text that cites the
    figure, nearby text).  ``require_crop`` skips pictures whose crop file was not written."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        page_texts = [page.get_text("text") for page in doc]
        records: list[dict[str, Any]] = []
        for page_result in result.pages:
            blocks = doc[page_result.page - 1].get_text("blocks") if page_result.figures else []
            for fig in page_result.figures:
                if require_crop and (not fig.crop_path or not Path(fig.crop_path).exists()):
                    continue
                number = caption_number(fig.caption)
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
                        "crop_path": fig.crop_path or "",
                        "mentions": _mentions(page_texts, number, fig.caption),
                        "nearby": _nearby_text(blocks, fig.bbox, fig.caption_bbox),
                        "description": "",
                    }
                )
    finally:
        doc.close()
    return records


def extract_figures(
    pdf_path: Path,
    doc_id: str,
    filename: str,
    settings: Settings,
    structure: DocumentStructure | None = None,
) -> list[dict[str, Any]]:
    """Detect pictures, write crops and build the searchable text. Blocking (CPU) work."""

    from backend.ingestion.Visualizing import detect_figures, load_vision_config, save_crops

    out_dir = settings.figures_root / doc_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    cfg = load_vision_config()
    result = detect_figures(pdf_path, cfg)
    if not result.figures:
        return []
    save_crops(pdf_path, result, out_dir, cfg)

    records = figure_records(pdf_path, result, doc_id, require_crop=True)

    # Which chapter / section is each picture in?  (Optional: never lose a picture over it.)
    if records and settings.structure.enabled:
        try:
            if structure is None:
                structure = safe_extract_structure(pdf_path, options_from_settings(settings.structure))
            if structure is not None:
                annotate_figures(structure, records)
        except Exception:  # noqa: BLE001
            logger.exception("Section lookup failed for %s; images are indexed without it", filename)
    return records


def build_search_text(record: dict[str, Any], filename: str) -> str:
    parts = [
        KIND_WORDS.get(str(record.get("kind")), "figure image picture"),
        str(record.get("caption") or ""),
        str(record.get("description") or ""),
        f"Referred to in the text: {record['mentions']}" if record.get("mentions") else "",
        f"Nearby text: {record['nearby']}" if record.get("nearby") else "",
        f"Page {record.get('page')} of {filename}",
        f"Section: {record['section_path']}" if record.get("section_id") not in (None, "", "s000") and record.get("section_path") else "",
    ]
    return ". ".join(part.strip() for part in parts if part and part.strip())


async def ingest_figures(
    file_path: Path,
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
    structure: DocumentStructure | None = None,
) -> int:
    """(Re)index every picture of one PDF. Safe to call repeatedly. Returns the number indexed."""
    resolved = settings or get_settings()
    if not resolved.figures.enabled or file_path.suffix.lower() != ".pdf":
        return 0
    target = store or get_store()

    records = await asyncio.to_thread(extract_figures, file_path, doc_id, filename, resolved, structure)
    target.delete_figures(doc_id)  # drop stale entries from an earlier scan
    target.delete_figure_chunks(doc_id)
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

    # Also make each description findable by ordinary questions ("how many children are shown?").
    if resolved.figures.index_descriptions:
        await _index_description_chunks(records, doc_id, filename, resolved, target)
    logger.info("Indexed %d image(s) from %s", len(records), filename)
    return len(records)


DESCRIPTION_CHUNK_BASE = 900_000  # chunk_index range reserved for image descriptions (text chunks use 0..n)


async def _index_description_chunks(
    records: list[dict[str, Any]],
    doc_id: str,
    filename: str,
    settings: Settings,
    store: ChromaStore,
) -> None:
    """Store each vision-model description as a text chunk tagged ``element="figure"``.

    Never raises: a failure here leaves the picture records (and normal text) intact.
    """
    chunks: list[dict[str, Any]] = []
    for record in records:
        description = str(record.get("description") or "").strip()
        if not description:
            continue
        caption = str(record.get("caption") or "").strip()
        text = f"[Page {record['page']} - Figure {record['figure_id']}]\n" + (f"{caption}\n" if caption else "") + description
        chunk: dict[str, Any] = {
            "text": text,
            "page": int(record["page"]),
            "chunk_index": DESCRIPTION_CHUNK_BASE + len(chunks),
            "element": "figure",
            "figure_id": record["figure_id"],
        }
        for key in ("section_id", "section_title", "section_path", "block_idx"):
            if record.get(key) is not None:
                chunk[key] = record[key]
        chunks.append(chunk)
    if not chunks:
        return
    try:
        embeddings = await embed([c["text"] for c in chunks], settings.embeddings)
        store.add_chunks(doc_id, filename, chunks, embeddings)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        logger.exception("Could not index image descriptions for %s as text", filename)

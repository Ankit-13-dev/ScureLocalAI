from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from backend.config import Settings, get_settings
from backend.ingestion.chunker import chunk_text
from backend.ingestion.figures import ingest_figures
from backend.ingestion.parsers import parse_docx, parse_pdf, parse_pptx, parse_text
from backend.ingestion.structure import DocumentStructure, annotate_chunks, options_from_settings, safe_extract_structure
from backend.retrieval.embedder import embed
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".txt": parse_text,
    ".md": parse_text,
}


async def ingest_document(
    file_path: Path,
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
) -> int:
    resolved = settings or get_settings()
    suffix = file_path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    pages = parser(file_path)
    chunks = chunk_text(
        pages,
        chunk_size=resolved.retrieval.chunk_size,
        overlap=resolved.retrieval.chunk_overlap,
    )

    # PDFs: learn the chapter / section layout so each chunk (and later each image) knows where
    # in the document it sits.  Optional: any failure leaves the chunks exactly as before.
    structure: DocumentStructure | None = None
    if suffix == ".pdf" and resolved.structure.enabled:
        structure = await asyncio.to_thread(safe_extract_structure, file_path, options_from_settings(resolved.structure))
        if structure is not None:
            try:
                stats = annotate_chunks(structure, chunks)
                logger.info("Sections for %s: %d found, %s", filename, len(structure.sections) - 1, stats)
            except Exception:  # noqa: BLE001
                logger.exception("Section annotation failed for %s; continuing without it", filename)

    embeddings = await embed([chunk["text"] for chunk in chunks], resolved.embeddings)
    target_store = store or get_store()
    target_store.add_chunks(doc_id, filename, chunks, embeddings)

    # Pictures are an add-on: if detection or embedding fails, text search must still work.
    if suffix == ".pdf":
        try:
            await ingest_figures(file_path, doc_id, filename, settings=resolved, store=target_store, structure=structure)
        except Exception:  # noqa: BLE001
            logger.exception("Image indexing failed for %s; the text was indexed normally", filename)
    return len(chunks)

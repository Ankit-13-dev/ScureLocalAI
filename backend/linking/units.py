"""The things that get linked: text chunks and figures of one PDF, each knowing its section.

``build_units`` runs exactly the ingestion steps (parse -> chunk -> section lookup, picture
detection -> section lookup) but touches no database and writes nothing, so evaluation tools
and later scoring code see the same data the chatbot indexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.config import Settings, get_settings
from backend.ingestion.chunker import chunk_text
from backend.ingestion.figures import figure_records
from backend.ingestion.parsers import parse_pdf
from backend.ingestion.structure import (
    DocumentStructure,
    annotate_chunks,
    annotate_figures,
    options_from_settings,
    extract_structure,
)
from backend.ingestion.Visualizing import detect_figures, load_vision_config


@dataclass
class TextUnit:
    unit_id: int
    text: str
    page: int
    section_id: str = "s000"
    section_path: str = ""
    block_idx: Optional[int] = None


@dataclass
class FigureUnit:
    figure_id: str
    page: int
    bbox: tuple
    kind: str = "unknown"
    caption: str = ""
    fig_number: Optional[int] = None
    section_id: str = "s000"
    section_path: str = ""
    block_idx: Optional[int] = None
    description: str = ""            # filled by the vision-model step (later phase)
    mentions: str = ""
    nearby: str = ""

    @property
    def text(self) -> str:
        """Everything known about the figure in words, for keyword / embedding comparison."""
        return " ".join(p for p in (self.caption, self.description, self.mentions, self.nearby) if p)


@dataclass
class DocumentUnits:
    path: str
    structure: DocumentStructure
    texts: list = field(default_factory=list)
    figures: list = field(default_factory=list)

    def text_by_id(self, unit_id: int) -> Optional[TextUnit]:
        return next((t for t in self.texts if t.unit_id == unit_id), None)


def build_units(pdf_path: Path, settings: Optional[Settings] = None) -> DocumentUnits:
    """Parse, chunk and locate everything in one PDF (blocking, CPU only)."""
    resolved = settings or get_settings()
    path = Path(pdf_path)

    structure = extract_structure(path, options_from_settings(resolved.structure))

    chunks = chunk_text(
        parse_pdf(path),
        chunk_size=resolved.retrieval.chunk_size,
        overlap=resolved.retrieval.chunk_overlap,
    )
    annotate_chunks(structure, chunks)
    texts = [
        TextUnit(
            unit_id=int(c["chunk_index"]),
            text=c["text"],
            page=int(c["page"]),
            section_id=c.get("section_id", "s000"),
            section_path=c.get("section_path", ""),
            block_idx=c.get("block_idx"),
        )
        for c in chunks
    ]

    result = detect_figures(path, load_vision_config())
    records = figure_records(path, result, doc_id="eval", require_crop=False)
    annotate_figures(structure, records)
    figures = [
        FigureUnit(
            figure_id=r["figure_id"],
            page=int(r["page"]),
            bbox=tuple(r["bbox"]),
            kind=r["kind"],
            caption=r["caption"],
            fig_number=r["fig_number"],
            section_id=r.get("section_id", "s000"),
            section_path=r.get("section_path", ""),
            block_idx=r.get("block_idx"),
            mentions=r["mentions"],
            nearby=r["nearby"],
        )
        for r in records
    ]
    return DocumentUnits(path=str(path), structure=structure, texts=texts, figures=figures)

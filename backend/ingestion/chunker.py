from __future__ import annotations

from typing import TypedDict


class _ChunkBase(TypedDict):
    text: str
    page: int
    chunk_index: int


class Chunk(_ChunkBase, total=False):
    """Optional fields are added for PDFs by structure.annotate_chunks()."""

    section_id: str
    section_title: str
    section_path: str
    section_ids: str      # all sections the chunk touches, best first (only when more than one)
    element: str          # "figure" for chunks that hold a vision-model image description
    figure_id: str
    section_source: str   # "matched" | "inherited"
    block_idx: int        # reading-order index of the block where the chunk starts
    y0: float


def chunk_text(pages: list[str], chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0
    step = chunk_size - overlap

    for page_number, page_text in enumerate(pages, start=1):
        text = " ".join(page_text.split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page_number, "chunk_index": chunk_index})
                chunk_index += 1
            if end == len(text):
                break
            start += step

    return chunks

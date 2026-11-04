"""structure.py - read a PDF's chapter / section structure and reading order.

Why this exists (Phase 1 of image<->text linking)
-------------------------------------------------
To decide which text an image belongs to, the system must know *where in the document*
both of them are.  "Same page" is not enough: a book has chapters, a paper has sections, and
two pages apart can still be the same topic while two paragraphs apart can be a new one.
This module recovers:

* the SECTION TREE (chapter > section > subsection) of a PDF,
* a global READING ORDER of text blocks (two-column pages handled),
* lookups:  text chunk -> section,   image bounding box -> section.

It is deterministic and offline (PyMuPDF only, no model).  Everything is best-effort: a PDF
with no detectable headings still gets a valid structure (one root section), so callers never
have to special-case failure.

How headings are found
----------------------
1. If the PDF has an outline / bookmarks (``get_toc``) that is trusted first - it is the
   author's own table of contents.
2. Otherwise headings are detected from typography: line size vs the document's body size,
   bold / italic / ALL-CAPS, numbering such as "2.3", "II.", "A.", "Chapter 4", short length,
   no trailing full stop, not a figure/table caption.  Running headers and footers (text that
   repeats on many pages in the page margins) and bare page numbers are ignored.
3. Levels come from the numbering depth when present, otherwise from the size ranking.

Reading order
-------------
Blocks are ordered top-to-bottom; on two-column pages the left column is read before the right
one, and full-width blocks (titles, wide figures) split the page into bands.

Command line
------------
    python backend/ingestion/structure.py paper.pdf            # print the outline
    python backend/ingestion/structure.py paper.pdf --json out.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # PyMuPDF >= 1.24
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

__all__ = [
    "Block",
    "Section",
    "DocumentStructure",
    "StructureOptions",
    "extract_structure",
    "safe_extract_structure",
    "options_from_settings",
    "annotate_chunks",
    "annotate_figures",
    "format_outline",
]

log = logging.getLogger("scurelocalai.structure")

Box = tuple  # (x0, y0, x1, y1) PDF points, origin top-left


# =============================================================================
# Options / data model
# =============================================================================
@dataclass(frozen=True)
class StructureOptions:
    use_outline: bool = True             # trust the PDF's own bookmarks when present
    max_heading_words: int = 16
    min_heading_size_ratio: float = 1.15  # heading font size / body font size
    max_levels: int = 4
    margin_ratio: float = 0.08           # top / bottom page share treated as header / footer zone


@dataclass
class Block:
    idx: int                 # global reading-order index over the whole document
    page: int                # 1-based
    order: int               # reading-order index within the page
    bbox: Box
    text: str
    size: float = 0.0
    bold: bool = False
    is_heading: bool = False


@dataclass
class Section:
    id: str                  # "s000" is the implicit root; real sections are s001, s002 ...
    title: str
    level: int               # 0 = root, 1 = chapter / top-level section, ...
    parent_id: Optional[str]
    path: tuple              # titles from the top-level ancestor down to this section
    page: int
    block_idx: int           # global index of the heading block (-1 for the root)
    source: str              # "outline" | "typography" | "root"

    @property
    def path_str(self) -> str:
        return " > ".join(self.path) if self.path else self.title

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "parent_id": self.parent_id,
            "path": list(self.path),
            "page": self.page,
            "block_idx": self.block_idx,
            "source": self.source,
        }


@dataclass
class _Line:
    block_no: int
    bbox: Box
    text: str
    size: float
    bold: float              # share of characters that are bold
    italic: float


class DocumentStructure:
    """Section tree + reading-ordered blocks for one PDF."""

    def __init__(self, path: str, page_count: int) -> None:
        self.path = path
        self.page_count = page_count
        self.title: str = ""
        self.method: str = "none"           # outline | typography | none
        self.body_size: float = 0.0
        self.sections: list = []
        self.blocks: list = []
        self.page_sizes: dict = {}           # page -> (width, height)
        self._page_blocks: dict = {}         # page -> [Block] in reading order
        self._layouts: dict = {}             # page -> (two_column, mid_x)
        self._section_starts: list = []
        self._norm: dict = {}                # page -> (normalised text, block start offsets)

    # ------------------------------------------------------------------ lookups
    @property
    def root(self) -> Section:
        return self.sections[0]

    def section_by_id(self, section_id: str) -> Optional[Section]:
        return next((s for s in self.sections if s.id == section_id), None)

    def section_at(self, block_idx: int) -> Section:
        """The section that contains the block with this global index."""
        pos = bisect.bisect_right(self._section_starts, block_idx) - 1
        return self.sections[max(0, pos)]

    def page_blocks(self, page: int) -> list:
        return self._page_blocks.get(page, [])

    def section_for_bbox(self, page: int, bbox: Box) -> tuple:
        """Section of an arbitrary box (e.g. a picture): the section of the last text block
        that precedes it in reading order.  Returns ``(Section, anchor_block_idx)``."""
        blocks = self.page_blocks(page)
        two_col, mid = self._layouts.get(page, (False, 0.0))
        width = self.page_sizes.get(page, (612.0, 792.0))[0]
        probe = _reading_keys([b.bbox for b in blocks] + [tuple(bbox)], width, two_col, mid)
        target = probe[-1]
        anchor = -1
        best_key = None
        for blk, key in zip(blocks, probe[:-1]):
            if key < target and (best_key is None or key > best_key):
                anchor, best_key = blk.idx, key
        if anchor < 0:
            # nothing precedes it on this page: it belongs to whatever section was open at the
            # end of the previous page
            first_here = blocks[0].idx if blocks else len(self.blocks)
            anchor = first_here - 1
        return self.section_at(anchor), anchor

    def locate_text(self, page: int, text: str) -> Optional[Block]:
        """Find the block where ``text`` starts (matching on alphanumerics only, so pypdf vs
        PyMuPDF whitespace / ligature differences do not matter)."""
        needle = _norm(text)
        if len(needle) < 12:
            return None
        norm_text, starts = self._norm_page(page)
        for probe in (needle[:48], needle[len(needle) // 2 : len(needle) // 2 + 48]):
            if len(probe) < 12:
                continue
            at = norm_text.find(probe)
            if at >= 0:
                blocks = self.page_blocks(page)
                return blocks[max(0, bisect.bisect_right(starts, at) - 1)]
        return None

    def section_weights(self, page: int, text: str) -> Optional[tuple]:
        """Where does ``text`` (a chunk) lie?  Returns ``(first_block, {section_id: share})``
        where share is the fraction of the chunk's characters falling in that section, or None
        if the text cannot be found on the page.  A chunk that crosses a heading belongs mostly
        to whichever side holds most of it."""
        needle = _norm(text)
        if len(needle) < 12:
            return None
        norm_text, starts = self._norm_page(page)
        tail = needle[-48:]
        best = None   # (length error, start, end)
        probe = needle[:48]
        at = norm_text.find(probe)
        tries = 0
        # the same sentence can occur twice on a page: take the occurrence whose end matches best
        while at >= 0 and tries < 25:
            ends = []
            pos = norm_text.find(tail, at)
            while pos >= 0 and len(ends) < 25:
                ends.append(pos + len(tail))
                pos = norm_text.find(tail, pos + 1)
            for end in ends or [at + len(needle)]:
                err = abs((end - at) - len(needle))
                if best is None or err < best[0]:
                    best = (err, at, end)
            at = norm_text.find(probe, at + 1)
            tries += 1
        if best is None:
            mid = needle[len(needle) // 2 : len(needle) // 2 + 48]
            pos = norm_text.find(mid) if len(mid) >= 12 else -1
            if pos < 0:
                return None
            start = max(0, pos - len(needle) // 2)   # the chunk began about half a chunk earlier
            best = (0, start, start + len(needle))
        _err, at, end = best
        blocks = self.page_blocks(page)
        first_i = max(0, bisect.bisect_right(starts, at) - 1)
        weights: dict = {}
        total = 0.0
        for i in range(first_i, len(blocks)):
            b_start = starts[i]
            b_end = starts[i + 1] if i + 1 < len(starts) else len(norm_text)
            if b_start >= end:
                break
            overlap = min(end, b_end) - max(at, b_start)
            if overlap > 0:
                sec = self.section_at(blocks[i].idx)
                weights[sec.id] = weights.get(sec.id, 0.0) + overlap
                total += overlap
        if not weights:
            return blocks[first_i], {self.section_at(blocks[first_i].idx).id: 1.0}
        return blocks[first_i], {k: v / total for k, v in weights.items()}

    def _norm_page(self, page: int) -> tuple:
        cached = self._norm.get(page)
        if cached is None:
            parts: list = []
            starts: list = []
            offset = 0
            for blk in self.page_blocks(page):
                starts.append(offset)
                normalised = _norm(blk.text)
                parts.append(normalised)
                offset += len(normalised)
            cached = ("".join(parts), starts)
            self._norm[page] = cached
        return cached

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "page_count": self.page_count,
            "title": self.title,
            "method": self.method,
            "body_font_size": round(self.body_size, 2),
            "section_count": len(self.sections) - 1,
            "block_count": len(self.blocks),
            "sections": [s.to_dict() for s in self.sections],
        }


# =============================================================================
# Small helpers
# =============================================================================
def _norm(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower() if ch.isalnum())


def _w(b: Box) -> float:
    return b[2] - b[0]


NUM_DECIMAL = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+\S")
NUM_ROMAN = re.compile(r"^([IVXLC]{1,6})[.)]\s+\S")
NUM_ALPHA = re.compile(r"^([A-Z])[.)]\s+\S")
KEYWORD = re.compile(r"^(chapter|section|part|unit|lesson|module|appendix|topic)\b", re.IGNORECASE)
CAPTION_START = re.compile(r"^(fig(?:ure)?|table|tab|plate|scheme|listing|algorithm|eq(?:uation)?)\b\.?\s*\d*", re.IGNORECASE)


def _numbering(text: str) -> tuple:
    """Return (kind, depth) for numbered headings, else (None, 0)."""
    m = NUM_DECIMAL.match(text)
    if m:
        return "decimal", m.group(1).count(".") + 1
    mr = NUM_ROMAN.match(text)
    if mr:
        rest = text.split(None, 1)[1] if " " in text else ""
        # "C." / "I." / "V." / "X." / "L." are ambiguous (Roman numeral or lettered subsection):
        # an ALL-CAPS title means a Roman-numbered section ("I. INTRODUCTION"), mixed case a
        # lettered subsection ("C. Input and Output").
        if len(mr.group(1)) == 1 and not (rest.isupper() or sum(c.isupper() for c in rest) > 0.8 * max(1, sum(c.isalpha() for c in rest))):
            return "alpha", 2
        return "roman", 1
    if KEYWORD.match(text):
        return "keyword", 1
    if NUM_ALPHA.match(text):
        return "alpha", 2
    return None, 0


# =============================================================================
# Reading order
# =============================================================================
def _page_layout(boxes: Sequence[Box], width: float) -> tuple:
    """Detect a two-column page.  Returns (two_column, mid_x)."""
    mid = width / 2.0
    narrow = [b for b in boxes if _w(b) < 0.55 * width]
    left = [b for b in narrow if (b[0] + b[2]) / 2 < mid]
    right = [b for b in narrow if (b[0] + b[2]) / 2 >= mid]
    if len(left) < 3 or len(right) < 3:
        return False, mid
    crossing = [b for b in narrow if b[0] < mid - 8 and b[2] > mid + 8]
    return (len(crossing) <= 0.15 * len(narrow)), mid


def _reading_keys(boxes: Sequence[Box], width: float, two_col: bool, mid: float) -> list:
    """Sortable key per box: (band, column order, y, x)."""
    if not two_col:
        return [(0, 0, b[1], b[0]) for b in boxes]
    kinds = []
    for b in boxes:
        full = _w(b) >= 0.6 * width or (b[0] < mid - 8 and b[2] > mid + 8)
        kinds.append(2 if full else (0 if (b[0] + b[2]) / 2 < mid else 1))
    fulls = [b for b, k in zip(boxes, kinds) if k == 2]
    keys = []
    for b, k in zip(boxes, kinds):
        band = sum(1 for f in fulls if f is not b and f[3] <= b[1] + 2)
        keys.append((band, k, b[1], b[0]))
    return keys


# =============================================================================
# Extraction
# =============================================================================
def _page_lines(page: "pymupdf.Page") -> list:
    lines: list = []
    try:
        data = page.get_text("dict")
    except Exception as exc:  # noqa: BLE001
        log.warning("text extraction failed on page %s: %s", page.number + 1, exc)
        return lines
    for block_no, blk in enumerate(data.get("blocks", [])):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            chars = sum(len(s["text"].strip()) for s in spans)
            dominant = max(spans, key=lambda s: len(s["text"].strip()))
            bold = sum(len(s["text"].strip()) for s in spans if (s["flags"] & 16) or "bold" in s["font"].lower())
            ital = sum(len(s["text"].strip()) for s in spans if (s["flags"] & 2) or "italic" in s["font"].lower() or "oblique" in s["font"].lower())
            text = "".join(s["text"] for s in ln["spans"]).strip()
            lines.append(
                _Line(
                    block_no=block_no,
                    bbox=tuple(float(v) for v in ln["bbox"]),
                    text=" ".join(text.split()),
                    size=float(dominant["size"]),
                    bold=bold / chars,
                    italic=ital / chars,
                )
            )
    return lines


def _repeated_margin_texts(per_page: dict, sizes: dict, opts: StructureOptions) -> set:
    """Normalised texts that repeat in the top / bottom margins of many pages (running heads)."""
    seen: dict = {}
    for page, lines in per_page.items():
        height = sizes[page][1]
        keys = set()
        for ln in lines:
            if ln.bbox[3] < opts.margin_ratio * height or ln.bbox[1] > (1 - opts.margin_ratio) * height:
                keys.add(re.sub(r"\d+", "#", ln.text.lower()))
        for k in keys:
            seen[k] = seen.get(k, 0) + 1
    need = max(3, int(0.3 * len(per_page)))
    return {k for k, c in seen.items() if c >= need}


def _is_margin_noise(ln: _Line, height: float, repeated: set, opts: StructureOptions) -> bool:
    in_margin = ln.bbox[3] < opts.margin_ratio * height or ln.bbox[1] > (1 - opts.margin_ratio) * height
    if not in_margin:
        return False
    if re.fullmatch(r"(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?", ln.text.lower()):
        return True
    return re.sub(r"\d+", "#", ln.text.lower()) in repeated


def _heading_candidate(ln: _Line, body: float, alone: bool, opts: StructureOptions) -> bool:
    text = ln.text
    words = len(text.split())
    if words == 0 or words > opts.max_heading_words or len(text) > 160:
        return False
    if re.fullmatch(r"[\d\W_]+", text) or CAPTION_START.match(text):
        return False
    if sum(c.isdigit() for c in text) > 0.4 * len(text):
        return False
    kind, _depth = _numbering(text)
    if text.endswith((".", ",", ";", ":")) and not (kind and words <= 2):
        return False
    ratio = ln.size / body if body else 1.0
    emph = ln.bold >= 0.9 or ln.italic >= 0.9
    letters = [c for c in text if c.isalpha()]
    upper = len(letters) >= 3 and all(c.isupper() for c in letters)

    if ratio >= opts.min_heading_size_ratio:
        return True
    if kind and (emph or upper or ratio >= 1.05):
        return True
    if ln.bold >= 0.9 and ratio >= 0.98 and words <= 10 and alone:
        return True
    if upper and words <= 8 and ratio >= 0.95 and alone and ln.bold >= 0.5:
        return True
    return False


def _split_blocks(page_no: int, lines: list, body: float, opts: StructureOptions) -> list:
    """Group lines into paragraph-like blocks, splitting headings away from body text."""
    by_block: dict = {}
    for ln in lines:
        by_block.setdefault(ln.block_no, []).append(ln)

    out: list = []  # (bbox, text, size, bold, is_heading)
    for _no, group in sorted(by_block.items()):
        flags = [_heading_candidate(ln, body, len(group) == 1, opts) for ln in group]
        run: list = []
        run_flag = None
        for ln, flag in zip(group, flags):
            same = run and flag == run_flag and (not flag or abs(ln.size - run[-1].size) < 0.6)
            if run and not same:
                out.append(_make_block(run, run_flag))
                run = []
            run.append(ln)
            run_flag = flag
        if run:
            out.append(_make_block(run, run_flag))
    return out


_CONNECTIVES = frozenset("of and the for to in with on by a an or as from at into via".split())


def _merge_adjacent_headings(pieces: list) -> list:
    """Join heading pieces that are really one wrapped heading, then drop pieces whose text
    cannot be a heading (arrows, formulas, lone symbols)."""
    ordered = sorted(pieces, key=lambda p: (p[0][1], p[0][0]))
    merged: list = []
    for piece in ordered:
        bbox, text, size, bold, is_head = piece
        if merged and not is_head and merged[-1][4]:
            pb, ptext, psize, pbold, _ = merged[-1]
            gap = bbox[1] - pb[3]
            overlap = min(bbox[2], pb[2]) - max(bbox[0], pb[0])
            dangling = ptext.rstrip().split()[-1].lower() in _CONNECTIVES
            if dangling and len(text.split()) <= 5 and -2 <= gap <= 0.9 * max(size, psize) and overlap > 0 and abs(size - psize) < 1.5:
                merged[-1] = (
                    (min(bbox[0], pb[0]), pb[1], max(bbox[2], pb[2]), bbox[3]),
                    f"{ptext} {text}".strip(), psize, pbold, True,
                )
                continue
        if merged and is_head and merged[-1][4]:
            pb, ptext, psize, pbold, _ = merged[-1]
            gap = bbox[1] - pb[3]
            overlap = min(bbox[2], pb[2]) - max(bbox[0], pb[0])
            if (
                -2 <= gap <= 0.9 * max(size, psize)
                and overlap > 0
                and abs(size - psize) < 0.6
                and _numbering(text)[0] is None
            ):
                merged[-1] = (
                    (min(bbox[0], pb[0]), pb[1], max(bbox[2], pb[2]), bbox[3]),
                    f"{ptext} {text}".strip(), psize, pbold and bold, True,
                )
                continue
        merged.append(piece)
    result: list = []
    for bbox, text, size, bold, is_head in merged:
        if is_head and not _plausible_heading_text(text):
            is_head = False
        result.append((bbox, text, size, bold, is_head))
    return result


def _plausible_heading_text(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    if letters < 3 or letters < 0.5 * len(text.replace(" ", "")):
        return False
    return not any(ch in text for ch in "→←⇒=<>∑∫√≈≤≥|{}")


def _make_block(run: list, is_heading: bool) -> tuple:
    text = ""
    for ln in run:
        if text.endswith("-") and ln.text[:1].islower():
            text = text[:-1] + ln.text
        else:
            text = (text + " " + ln.text).strip()
    x0 = min(l.bbox[0] for l in run)
    y0 = min(l.bbox[1] for l in run)
    x1 = max(l.bbox[2] for l in run)
    y1 = max(l.bbox[3] for l in run)
    size = sum(l.size for l in run) / len(run)
    bold = sum(l.bold for l in run) / len(run) >= 0.9
    return ((x0, y0, x1, y1), text, size, bold, is_heading)


def extract_structure(pdf_path: Any, options: Optional[StructureOptions] = None) -> DocumentStructure:
    """Build the :class:`DocumentStructure` of a PDF.  Never raises for odd content; a document
    without recognisable headings yields a single root section."""
    opts = options or StructureOptions()
    doc = pymupdf.open(str(pdf_path))
    try:
        st = DocumentStructure(str(pdf_path), len(doc))

        per_page: dict = {}
        for i, page in enumerate(doc):
            if page.rotation:
                page.set_rotation(0)
            st.page_sizes[i + 1] = (float(page.rect.width), float(page.rect.height))
            per_page[i + 1] = _page_lines(page)

        # body font size = the size carrying the most characters
        counts: Counter = Counter()
        for lines in per_page.values():
            for ln in lines:
                counts[round(ln.size, 1)] += len(ln.text)
        st.body_size = float(counts.most_common(1)[0][0]) if counts else 0.0

        repeated = _repeated_margin_texts(per_page, st.page_sizes, opts) if len(per_page) >= 3 else set()

        # blocks per page (+ heading flags), then reading order
        raw_headings: list = []   # (global idx, text, size, bold, italic?, numbering)
        for page_no in sorted(per_page):
            height = st.page_sizes[page_no][1]
            lines = [l for l in per_page[page_no] if not _is_margin_noise(l, height, repeated, opts)]
            pieces = _split_blocks(page_no, lines, st.body_size, opts)
            pieces = _merge_adjacent_headings(pieces)
            width = st.page_sizes[page_no][0]
            two_col, mid = _page_layout([p[0] for p in pieces], width)
            st._layouts[page_no] = (two_col, mid)
            keys = _reading_keys([p[0] for p in pieces], width, two_col, mid)
            ordered = [p for _k, p in sorted(zip(keys, pieces), key=lambda kp: kp[0])]
            page_blocks: list = []
            for order, (bbox, text, size, bold, is_head) in enumerate(ordered):
                blk = Block(
                    idx=len(st.blocks), page=page_no, order=order, bbox=bbox, text=text,
                    size=size, bold=bold, is_heading=is_head,
                )
                st.blocks.append(blk)
                page_blocks.append(blk)
            st._page_blocks[page_no] = page_blocks

        _find_title(st)

        used_outline = False
        if opts.use_outline:
            try:
                toc = doc.get_toc(simple=True)
            except Exception:  # noqa: BLE001
                toc = []
            if len([t for t in toc if str(t[1]).strip()]) >= 2:
                used_outline = _sections_from_outline(st, toc, opts)

        if not used_outline:
            _sections_from_typography(st, opts)
        _finalise_sections(st)
        return st
    finally:
        doc.close()


def _find_title(st: DocumentStructure) -> None:
    first = st.page_blocks(1)
    if not first or not st.body_size:
        return
    height = st.page_sizes[1][1]
    top = [b for b in first if b.bbox[1] < 0.45 * height and len(b.text.split()) <= 30]
    if not top:
        return
    biggest = max(top, key=lambda b: b.size)
    if biggest.size < 1.4 * st.body_size:
        return
    if _numbering(biggest.text)[0] is not None:
        return  # "1 Introduction" / "Chapter 1" is a section, not the document title
    parts = sorted(
        (b for b in top if abs(b.size - biggest.size) < 0.6 and abs(b.bbox[1] - biggest.bbox[1]) < 120),
        key=lambda b: (b.bbox[1], b.bbox[0]),
    )
    part_ids = {b.idx for b in parts}
    # the same large style used again later (chapter headings) means it is a heading style, not a title
    if any(b.idx not in part_ids and abs(b.size - biggest.size) < 0.6 and b.is_heading for b in st.blocks):
        return
    st.title = " ".join(b.text for b in parts)
    st._title_blocks = {b.idx for b in parts}  # type: ignore[attr-defined]
    for b in parts:
        b.is_heading = False  # the document title is not a section


def _root(st: DocumentStructure) -> Section:
    return Section("s000", st.title or "Document start", 0, None, (), 1, -1, "root")


def _sections_from_typography(st: DocumentStructure, opts: StructureOptions) -> None:
    heads = [b for b in st.blocks if b.is_heading and b.idx not in getattr(st, "_title_blocks", ())]
    if not heads:
        st.sections = [_root(st)]
        st.method = "none"
        return
    styles = sorted({(round(b.size, 1), b.bold) for b in heads}, key=lambda s: (-s[0], not s[1]))
    rank = {s: i + 1 for i, s in enumerate(styles)}

    depths = []
    n_numbered = 0
    for b in heads:
        kind, depth = _numbering(b.text)
        if kind in ("decimal", "roman"):
            n_numbered += 1
        depths.append((kind, depth))
    numbered_share = sum(1 for kind, _d in depths if kind in ("decimal", "roman", "alpha")) / len(heads)
    use_numbers = numbered_share >= 0.3 and n_numbered > 0

    sections = [_root(st)]
    stack: list = [sections[0]]
    for b, (kind, depth) in zip(heads, depths):
        style_rank = rank[(round(b.size, 1), b.bold)]
        if use_numbers:
            # numbered document: numbering decides; un-numbered headings (Abstract, References,
            # Chapter ...) are top-level
            level = depth if kind in ("decimal", "roman", "alpha") else 1
        else:
            level = style_rank
        level = max(1, min(level, opts.max_levels))
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else sections[0]
        level = min(level, parent.level + 1)  # no skipped levels
        sec = Section(
            id=f"s{len(sections):03d}", title=b.text, level=level, parent_id=parent.id if parent.level > 0 else None,
            path=parent.path + (b.text,), page=b.page, block_idx=b.idx, source="typography",
        )
        sections.append(sec)
        stack.append(sec)
    st.sections = sections
    st.method = "typography"


def _sections_from_outline(st: DocumentStructure, toc: list, opts: StructureOptions) -> bool:
    entries = []
    for level, title, page in toc:
        title = " ".join(str(title).split())
        if not title or not (1 <= int(page) <= st.page_count):
            continue
        entries.append((int(level), title, int(page)))
    if len(entries) < 2:
        return False

    located = []
    for level, title, page in entries:
        key = _norm(title)[:30]
        idx = None
        for blk in st.page_blocks(page):
            if key and key in _norm(blk.text)[: len(key) + 20]:
                idx = blk.idx
                blk.is_heading = True
                break
        if idx is None:
            blocks = st.page_blocks(page)
            idx = blocks[0].idx if blocks else len(st.blocks)
        located.append((idx, level, title, page))
    located.sort(key=lambda t: t[0])  # reading order, not bookmark order

    sections = [_root(st)]
    stack: list = [sections[0]]
    for idx, level, title, page in located:
        level = max(1, min(level, opts.max_levels))
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else sections[0]
        level = min(level, parent.level + 1)
        sec = Section(
            id=f"s{len(sections):03d}", title=title, level=level, parent_id=parent.id if parent.level > 0 else None,
            path=parent.path + (title,), page=page, block_idx=idx, source="outline",
        )
        sections.append(sec)
        stack.append(sec)
    st.sections = sections
    st.method = "outline"
    return True


def _finalise_sections(st: DocumentStructure) -> None:
    st.sections.sort(key=lambda s: s.block_idx)  # root (-1) stays first
    st._section_starts = [s.block_idx for s in st.sections]


def safe_extract_structure(pdf_path: Any, options: Optional[StructureOptions] = None) -> Optional[DocumentStructure]:
    """Like :func:`extract_structure` but returns None (and logs) instead of raising."""
    try:
        return extract_structure(pdf_path, options)
    except Exception:  # noqa: BLE001 - structure is an enhancement, never a reason to fail an upload
        log.exception("Structure extraction failed for %s", pdf_path)
        return None


def options_from_settings(settings: Any) -> StructureOptions:
    """Build options from ``backend.config.StructureSettings`` (duck-typed, no import cycle)."""
    return StructureOptions(
        use_outline=bool(settings.use_outline),
        max_heading_words=int(settings.max_heading_words),
        min_heading_size_ratio=float(settings.min_heading_size_ratio),
        max_levels=int(settings.max_levels),
    )


# =============================================================================
# Lookups used by the ingestion pipeline
# =============================================================================
def annotate_chunks(structure: DocumentStructure, chunks: list) -> dict:
    """Add section / position fields to text chunks (in place).

    Fields added: ``section_id`` / ``section_title`` / ``section_path`` (the section holding most
    of the chunk), ``section_ids`` (every section the chunk touches, comma separated, only when
    more than one), ``block_idx`` / ``y0`` (where the chunk starts) and ``section_source``
    ("matched" when the chunk's text was found on the page, "inherited" when it was assumed to
    continue the previous chunk's section).  Returns ``{"total", "matched", "inherited"}``.
    """
    matched = inherited = 0
    last: Optional[Section] = None
    last_idx = -1
    for chunk in chunks:
        page = int(chunk["page"])
        found = None
        for candidate_page in (page, page - 1, page + 1):
            if 1 <= candidate_page <= structure.page_count:
                found = structure.section_weights(candidate_page, chunk["text"])
                if found:
                    break
        if found:
            blk, weights = found
            main_id = max(weights, key=lambda k: weights[k])
            sec = structure.section_by_id(main_id) or structure.section_at(blk.idx)
            chunk.update(
                section_id=sec.id, section_title=sec.title, section_path=sec.path_str,
                block_idx=blk.idx, y0=round(float(blk.bbox[1]), 1), section_source="matched",
            )
            if len(weights) > 1:
                chunk["section_ids"] = ",".join(k for k, _v in sorted(weights.items(), key=lambda kv: -kv[1]))
            last, last_idx = sec, blk.idx
            matched += 1
        elif last is not None:
            chunk.update(
                section_id=last.id, section_title=last.title, section_path=last.path_str,
                block_idx=last_idx, section_source="inherited",
            )
            inherited += 1
    return {"total": len(chunks), "matched": matched, "inherited": inherited}


def annotate_figures(structure: DocumentStructure, records: list) -> None:
    """Add the section and reading-order position to figure records (in place)."""
    for rec in records:
        try:
            sec, anchor = structure.section_for_bbox(int(rec["page"]), tuple(rec["bbox"]))
        except Exception:  # noqa: BLE001 - never lose a figure because of its section
            log.exception("section lookup failed for %s", rec.get("figure_id"))
            continue
        rec.update(section_id=sec.id, section_title=sec.title, section_path=sec.path_str, block_idx=int(anchor))


def format_outline(structure: DocumentStructure) -> str:
    lines = [f"{structure.path}  ({structure.page_count} pages, method={structure.method}, body font={structure.body_size:.1f}pt)"]
    if structure.title:
        lines.append(f"title: {structure.title}")
    for sec in structure.sections[1:]:
        lines.append(f"{'  ' * (sec.level - 1)}{sec.id}  [{sec.level}] {sec.title}  (p.{sec.page})")
    if len(structure.sections) == 1:
        lines.append("(no headings detected - one root section)")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Print the detected chapter / section outline of a PDF.")
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("--json", help="write the structure to this JSON file (single PDF only)")
    args = ap.parse_args(argv)
    status = 0
    for item in args.pdf:
        path = Path(item)
        if not path.is_file():
            print(f"File not found: {path}")
            status = 2
            continue
        try:
            st = extract_structure(path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed on {path.name}: {exc}")
            status = 1
            continue
        print(format_outline(st))
        if args.json and len(args.pdf) == 1:
            Path(args.json).write_text(json.dumps(st.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"structure written to {args.json}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

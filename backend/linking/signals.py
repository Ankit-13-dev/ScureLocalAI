"""Individual pieces of evidence that an image and a piece of text belong together.

Each signal returns a number in [0, 1] and is independently testable.  They are deliberately
simple and explainable; a later phase combines them (and adds vision-model descriptions).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

from backend.ingestion.structure import DocumentStructure
from backend.linking.units import FigureUnit, TextUnit


_FIG_PREFIX = re.compile(
    r"\bfig(?:ure)?s?\.?\s*(\(?\d{1,3}[a-z]?\)?(?:\s*(?:,|and|&|to|-|\u2013)\s*\(?\d{1,3}[a-z]?\)?)*)",
    re.IGNORECASE,
)
_NUM_TOKEN = re.compile(r"(\d{1,3})")
_RANGE_WORDS = ("-", "\u2013", "to")


def cited_numbers(text: str) -> set:
    """Figure numbers cited in ``text``: "Fig. 3", "Figures 2 and 4", "Figs. 5-7", "(Fig. 3a)"."""
    cited: set = set()
    for match in _FIG_PREFIX.finditer(text):
        seq = match.group(1)
        nums = [int(n) for n in _NUM_TOKEN.findall(seq)]
        cited.update(nums)
        if len(nums) == 2 and any(w in seq for w in _RANGE_WORDS) and 0 < nums[1] - nums[0] <= 10:
            cited.update(range(nums[0], nums[1] + 1))
    return cited


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def contains_caption(fig: FigureUnit, unit: TextUnit) -> bool:
    """True when this text chunk is (or contains) the figure's own caption."""
    head = _norm(fig.caption)[:40]
    return len(head) >= 12 and head in _norm(unit.text)


def reference_score(fig: FigureUnit, unit: TextUnit) -> float:
    """1.0 when the text cites the figure by number ("as shown in Fig. 3") - the strongest
    evidence there is.  The chunk holding the caption itself does not count as a citation."""
    if fig.fig_number is None or contains_caption(fig, unit):
        return 0.0
    return 1.0 if fig.fig_number in cited_numbers(unit.text) else 0.0


def proximity_score(fig: FigureUnit, unit: TextUnit, scale: float = 6.0) -> float:
    """Closeness in reading order (blocks apart); falls back to pages apart if positions are unknown."""
    if fig.block_idx is not None and unit.block_idx is not None:
        return math.exp(-abs(fig.block_idx - unit.block_idx) / scale)
    return math.exp(-abs(fig.page - unit.page) * 2.0)


def section_score(fig: FigureUnit, unit: TextUnit, structure: Optional[DocumentStructure]) -> float:
    """1.0 same section, 0.5 neighbouring section, 0.15 same chapter, else 0."""
    if fig.section_id == unit.section_id:
        return 1.0
    if structure is None:
        return 0.0
    ids = [s.id for s in structure.sections]
    if fig.section_id in ids and unit.section_id in ids:
        if abs(ids.index(fig.section_id) - ids.index(unit.section_id)) == 1:
            return 0.5
        a, b = structure.section_by_id(fig.section_id), structure.section_by_id(unit.section_id)
        if a and b and a.path[:1] and a.path[:1] == b.path[:1]:
            return 0.15
    return 0.0


_STOP = frozenset(
    "the a an of in on at to for from with and or but is are was were be been this that these those it its as by "
    "we our can will not no also than then there here which where when into over under between using used use "
    "fig figure table page shown show shows".split()
)


def tokens(text: str) -> list:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words if len(w) >= 3 and w not in _STOP]


def build_idf(texts: Iterable[str]) -> dict:
    """Inverse document frequency over the document's own chunks (rare words weigh more)."""
    docs = [set(tokens(t)) for t in texts]
    n = max(1, len(docs))
    df: Counter = Counter()
    for d in docs:
        df.update(d)
    return {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}


def keyword_score(fig_text: str, unit_text: str, idf: dict) -> float:
    """TF-IDF cosine between the figure's words (caption, nearby text, ...) and a chunk."""
    a, b = Counter(tokens(fig_text)), Counter(tokens(unit_text))
    if not a or not b:
        return 0.0
    wa = {w: c * idf.get(w, 1.0) for w, c in a.items()}
    wb = {w: c * idf.get(w, 1.0) for w, c in b.items()}
    dot = sum(wa[w] * wb[w] for w in wa.keys() & wb.keys())
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    return dot / (na * nb) if na and nb else 0.0

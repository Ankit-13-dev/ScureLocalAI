"""Linkers: given a figure, rank the text chunks that most likely discuss it.

``NearestTextLinker`` is the baseline every smarter method must beat: it only looks at how
close a chunk is to the picture.  ``SignalLinker`` is a first, hand-weighted combination of the
signals in ``signals.py`` (no vision model yet); its weights are fitted in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.linking import signals
from backend.linking.units import DocumentUnits, FigureUnit


@dataclass
class Link:
    unit_id: int
    section_id: str
    score: float
    evidence: dict = field(default_factory=dict)


class NearestTextLinker:
    name = "baseline"

    def __init__(self, units: DocumentUnits) -> None:
        self.units = units

    def rank(self, fig: FigureUnit, top_k: int = 5) -> list:
        links = []
        for unit in self.units.texts:
            prox = signals.proximity_score(fig, unit)
            # a different page always ranks below the same page
            page_gap = abs(unit.page - fig.page)
            links.append((page_gap, -prox, Link(unit.unit_id, unit.section_id, prox, {"proximity": round(prox, 3)})))
        links.sort(key=lambda t: (t[0], t[1], t[2].unit_id))
        return [l for _g, _p, l in links[:top_k]]


class SignalLinker:
    name = "signals"
    WEIGHTS = {"reference": 0.40, "keyword": 0.25, "proximity": 0.20, "section": 0.15}

    def __init__(self, units: DocumentUnits, weights: dict | None = None) -> None:
        self.units = units
        self.weights = {**self.WEIGHTS, **(weights or {})}
        self.idf = signals.build_idf(t.text for t in units.texts)

    def rank(self, fig: FigureUnit, top_k: int = 5) -> list:
        scored = []
        for unit in self.units.texts:
            parts = {
                "reference": signals.reference_score(fig, unit),
                "keyword": signals.keyword_score(fig.text, unit.text, self.idf),
                "proximity": signals.proximity_score(fig, unit),
                "section": signals.section_score(fig, unit, self.units.structure),
            }
            total = sum(self.weights[k] * v for k, v in parts.items())
            scored.append(Link(unit.unit_id, unit.section_id, total, {k: round(v, 3) for k, v in parts.items()}))
        scored.sort(key=lambda l: (-l.score, l.unit_id))
        return scored[:top_k]


LINKERS = {"baseline": NearestTextLinker, "signals": SignalLinker}

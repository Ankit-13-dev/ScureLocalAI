"""Score a linker against hand-labelled gold data (section level)."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.linking.units import DocumentUnits


@dataclass
class FigureResult:
    figure_id: str
    page: int
    relevant: frozenset
    no_related: bool
    predicted: str | None            # section of the top link, or None if the linker abstained
    top_score: float
    ranked_sections: list = field(default_factory=list)   # distinct sections, best first
    correct_top1: bool = False
    correct_top3: bool = False
    reciprocal_rank: float = 0.0


def is_labeled(gold_fig: dict) -> bool:
    return bool(gold_fig.get("relevant_sections")) or bool(gold_fig.get("no_related_text"))


def evaluate(units: DocumentUnits, gold_figures: list, linker, min_score: float = 0.0, top_k: int = 10) -> tuple:
    """Return (per-figure results, summary).  ``min_score`` is the confidence below which the
    linker is considered to say "not sure"."""
    by_id = {f.figure_id: f for f in units.figures}
    results: list = []
    missing: list = []
    unlabeled = 0
    for g in gold_figures:
        if not is_labeled(g):
            unlabeled += 1
            continue
        fig = by_id.get(g["figure_id"])
        if fig is None:
            missing.append(g["figure_id"])
            continue
        links = linker.rank(fig, top_k=top_k)
        confident = [l for l in links if l.score >= min_score]
        sections: list = []
        for l in links:
            if l.section_id not in sections:
                sections.append(l.section_id)
        relevant = frozenset(g.get("relevant_sections") or [])
        res = FigureResult(
            figure_id=fig.figure_id,
            page=fig.page,
            relevant=relevant,
            no_related=bool(g.get("no_related_text")),
            predicted=confident[0].section_id if confident else None,
            top_score=links[0].score if links else 0.0,
            ranked_sections=sections,
        )
        if relevant:
            res.correct_top1 = res.predicted in relevant
            res.correct_top3 = any(l.section_id in relevant for l in confident[:3])
            for rank, sec in enumerate(sections, start=1):
                if sec in relevant:
                    res.reciprocal_rank = 1.0 / rank
                    break
        results.append(res)

    with_text = [r for r in results if r.relevant]
    without = [r for r in results if r.no_related and not r.relevant]
    n = len(with_text)
    summary = {
        "labeled": len(results),
        "unlabeled_skipped": unlabeled,
        "gold_not_detected": missing,
        "with_related_text": n,
        "top1_accuracy": sum(r.correct_top1 for r in with_text) / n if n else None,
        "top3_accuracy": sum(r.correct_top3 for r in with_text) / n if n else None,
        "mrr": sum(r.reciprocal_rank for r in with_text) / n if n else None,
        "wrong_link_rate": sum(1 for r in with_text if r.predicted is not None and not r.correct_top1) / n if n else None,
        "not_sure_rate": sum(1 for r in with_text if r.predicted is None) / n if n else None,
        "no_text_figures": len(without),
        "correct_not_sure_on_no_text": (sum(1 for r in without if r.predicted is None) / len(without)) if without else None,
    }
    return results, summary


def format_summary(name: str, summary: dict) -> str:
    def pct(v):
        return "  n/a" if v is None else f"{100 * v:5.1f}%"

    return (
        f"{name:<10} labeled={summary['labeled']:<3} "
        f"top1={pct(summary['top1_accuracy'])} top3={pct(summary['top3_accuracy'])} "
        f"MRR={'n/a' if summary['mrr'] is None else f'{summary['mrr']:.2f}'} "
        f"wrong={pct(summary['wrong_link_rate'])} not-sure={pct(summary['not_sure_rate'])} "
        f"no-text-correct={pct(summary['correct_not_sure_on_no_text'])}"
    )

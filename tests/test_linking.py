import json
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("cv2")

from backend.linking import signals  # noqa: E402
from backend.linking.baseline import LINKERS, NearestTextLinker, SignalLinker  # noqa: E402
from backend.linking.evaluate import evaluate  # noqa: E402
from backend.linking.units import FigureUnit, TextUnit, build_units  # noqa: E402
from tests.pdf_builders import make_figure_book  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("as shown in Fig. 3, the", {3}),
        ("see Figures 2 and 4 for", {2, 4}),
        ("Figs. 5-7 illustrate", {5, 6, 7}),
        ("(Fig. 3a) and Fig 10", {3, 10}),
        ("no citation here", set()),
    ],
)
def test_cited_numbers(text, expected) -> None:
    assert signals.cited_numbers(text) == expected


def _fig(**kw) -> FigureUnit:
    base = dict(figure_id="p001_f01", page=1, bbox=(0, 0, 10, 10), caption="Fig. 1. Glacier retreat photograph.", fig_number=1,
                section_id="s001", block_idx=10)
    base.update(kw)
    return FigureUnit(**base)


def test_citation_counts_but_the_caption_chunk_does_not() -> None:
    fig = _fig()
    cites = TextUnit(0, "As shown in Fig. 1, the glacier front retreated.", 2, "s002", block_idx=30)
    caption = TextUnit(1, "Fig. 1. Glacier retreat photograph. Next paragraph.", 1, "s001", block_idx=11)

    assert signals.reference_score(fig, cites) == 1.0
    assert signals.reference_score(fig, caption) == 0.0
    assert signals.reference_score(_fig(fig_number=None), cites) == 0.0


def test_proximity_decays_with_distance_and_falls_back_to_pages() -> None:
    fig = _fig()
    near = TextUnit(0, "x", 1, block_idx=11)
    far = TextUnit(1, "x", 1, block_idx=60)
    assert signals.proximity_score(fig, near) > signals.proximity_score(fig, far)
    assert signals.proximity_score(_fig(block_idx=None), TextUnit(2, "x", 3)) < signals.proximity_score(_fig(block_idx=None), TextUnit(3, "x", 1))


def test_keyword_score_prefers_shared_rare_words() -> None:
    texts = ["glacier retreat measured by photographs", "temperature calibration of sensors", "budget and schedule"]
    idf = signals.build_idf(texts)
    fig_text = "Glacier retreat photograph"
    assert signals.keyword_score(fig_text, texts[0], idf) > signals.keyword_score(fig_text, texts[1], idf)
    assert signals.keyword_score("", texts[0], idf) == 0.0


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    pdf = make_figure_book(tmp_path_factory.mktemp("fb") / "figbook.pdf")
    return build_units(pdf)


def test_build_units_places_figure_in_intro_and_text_in_results(units) -> None:
    assert len(units.figures) == 1
    fig = units.figures[0]
    assert fig.fig_number == 1 and "Introduction" in fig.section_path
    sections = {t.section_path.split(" > ")[-1] for t in units.texts}
    assert {"1 Introduction", "2 Results"} <= sections


def test_signals_beat_nearest_text_when_the_figure_is_discussed_elsewhere(units) -> None:
    fig = units.figures[0]
    base_top = NearestTextLinker(units).rank(fig, 1)[0]
    sig_top = SignalLinker(units).rank(fig, 1)[0]

    assert base_top.section_id == fig.section_id                       # baseline: the nearby intro text
    results_id = next(s.id for s in units.structure.sections if s.title == "2 Results")
    assert sig_top.section_id == results_id                            # signals: the text that cites Fig. 1
    assert sig_top.evidence["reference"] == 1.0

    gold = [{"figure_id": fig.figure_id, "relevant_sections": [results_id]}]
    _r, base = evaluate(units, gold, NearestTextLinker(units))
    _r, sig = evaluate(units, gold, SignalLinker(units))
    assert base["top1_accuracy"] == 0.0 and sig["top1_accuracy"] == 1.0


def test_evaluate_handles_unlabeled_missing_and_no_text_figures(units) -> None:
    fig = units.figures[0]
    gold = [
        {"figure_id": fig.figure_id, "no_related_text": True},
        {"figure_id": "p009_f09", "relevant_sections": ["s001"]},   # no longer detected
        {"figure_id": "p001_f99"},                                    # unlabeled
    ]
    results, summary = evaluate(units, gold, SignalLinker(units), min_score=0.99)

    assert summary["labeled"] == 1 and summary["unlabeled_skipped"] == 1
    assert summary["gold_not_detected"] == ["p009_f09"]
    assert summary["correct_not_sure_on_no_text"] == 1.0
    assert results[0].predicted is None


def test_linker_registry() -> None:
    assert set(LINKERS) == {"baseline", "signals"}


def test_gold_template_and_evaluation_tools_work_end_to_end(tmp_path: Path, capsys) -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pdf = make_figure_book(tmp_path / "figbook.pdf")
    gold_dir = tmp_path / "gold"
    assert load("make_gold_template").main([str(pdf), "--out", str(gold_dir)]) == 0
    gold_file = gold_dir / "figbook.gold.json"
    data = json.loads(gold_file.read_text(encoding="utf-8"))
    assert len(data["figures"]) == 1 and data["figures"][0]["relevant_sections"] == []
    assert (gold_dir / "figbook_review" / "crops" / f"{data['figures'][0]['figure_id']}.png").is_file()

    # unlabeled template: tool reports it instead of failing
    capsys.readouterr()
    assert load("evaluate_links").main([str(gold_file), "--pdf", str(pdf)]) == 0
    assert "nothing labelled yet" in capsys.readouterr().out

    results_id = next(s["id"] for s in data["sections"] if s["title"] == "2 Results")
    data["figures"][0]["relevant_sections"] = [results_id]
    gold_file.write_text(json.dumps(data), encoding="utf-8")
    assert load("evaluate_links").main([str(gold_file), "--pdf", str(pdf), "-v"]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "signals" in out and "100.0%" in out

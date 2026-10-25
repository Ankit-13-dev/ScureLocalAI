from pathlib import Path

from backend.ingestion.Visualizing import (
    VisionConfig,
    build_selftest_pdf,
    detect_figures,
    load_vision_config,
)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def test_detects_every_visual_and_ignores_text_tables_and_boxes(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    truth = build_selftest_pdf(pdf)
    result = detect_figures(pdf, VisionConfig())

    for page in result.pages:
        found = [f.bbox for f in page.figures]
        expected = [box for box, _kind in truth[page.page]]
        assert len(found) == len(expected), f"page {page.page}: {len(found)} found, {len(expected)} expected"
        for box in expected:
            assert max(_iou(box, f) for f in found) >= 0.5


def test_photo_and_scanned_page_are_recognised(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    build_selftest_pdf(pdf)
    result = detect_figures(pdf, VisionConfig())

    assert result.pages[1].figures[0].source == "embedded_image"
    assert result.pages[1].figures[0].kind == "photo"
    assert result.pages[2].is_scanned
    assert {f.source for f in result.pages[2].figures} == {"raster_layout"}


def test_figure_regions_are_json_serialisable(tmp_path: Path) -> None:
    import json

    pdf = tmp_path / "demo.pdf"
    build_selftest_pdf(pdf)
    payload = detect_figures(pdf, VisionConfig()).to_dict()

    assert payload["figure_count"] == 7
    json.dumps(payload)


def test_config_loads_without_vision_section(tmp_path: Path) -> None:
    empty = tmp_path / "config.yaml"
    empty.write_text("llm:\n  model: x\n", encoding="utf-8")
    assert load_vision_config(empty).crop_dpi == VisionConfig().crop_dpi

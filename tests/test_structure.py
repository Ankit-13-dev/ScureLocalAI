from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from backend.ingestion.chunker import chunk_text  # noqa: E402
from backend.ingestion.parsers import parse_pdf  # noqa: E402
from backend.ingestion.structure import (  # noqa: E402
    StructureOptions,
    annotate_chunks,
    annotate_figures,
    extract_structure,
    safe_extract_structure,
)
from tests.pdf_builders import make_book, make_plain, make_two_column  # noqa: E402


def titles(st):
    return [(s.title, s.level) for s in st.sections[1:]]


def test_numbered_headings_levels_and_running_text_ignored(tmp_path: Path) -> None:
    st = extract_structure(make_book(tmp_path / "book.pdf"))

    assert st.method == "typography"
    assert titles(st) == [
        ("1 Introduction", 1), ("1.1 Background", 2), ("1.2 Goals", 2), ("2 Methods", 1), ("2.1 Data", 2),
    ]
    assert not any("Running Title" in s.title or s.title.strip().isdigit() for s in st.sections)
    assert st.section_by_id("s003").path == ("1 Introduction", "1.2 Goals")


def test_pdf_outline_is_trusted_when_present(tmp_path: Path) -> None:
    st = extract_structure(make_book(tmp_path / "book.pdf", outline=True))

    assert st.method == "outline"
    assert titles(st)[0] == ("1 Introduction", 1)
    assert [s.source for s in st.sections[1:]] == ["outline"] * 5


def test_outline_can_be_switched_off(tmp_path: Path) -> None:
    st = extract_structure(make_book(tmp_path / "book.pdf", outline=True), StructureOptions(use_outline=False))
    assert st.method == "typography"


def test_document_without_headings_has_one_root_section(tmp_path: Path) -> None:
    pdf = make_plain(tmp_path / "plain.pdf")
    st = extract_structure(pdf)
    chunks = chunk_text(parse_pdf(pdf), 300, 30)
    stats = annotate_chunks(st, chunks)

    assert len(st.sections) == 1 and st.method == "none"
    assert stats["total"] == len(chunks)
    assert {c["section_id"] for c in chunks} == {"s000"}


def _chunk(page: int, text: str, index: int = 0) -> dict:
    return {"text": " ".join(text.split()), "page": page, "chunk_index": index}


def test_each_paragraph_chunk_maps_to_its_own_section(tmp_path: Path) -> None:
    from tests.pdf_builders import LONG

    st = extract_structure(make_book(tmp_path / "book.pdf"))
    expect = [
        (1, "glaciers and meltwater hydrology", "1 Introduction"),
        (1, "the history of ice core sampling", "1.1 Background"),
        (1, "the research goals and hypotheses", "1.2 Goals"),
        (2, "the sampling methods and calibration", "2 Methods"),
        (2, "the measured temperature dataset", "2.1 Data"),
    ]
    chunks = [_chunk(page, LONG.format(topic=topic) * 2, i) for i, (page, topic, _t) in enumerate(expect)]
    stats = annotate_chunks(st, chunks)

    assert stats == {"total": 5, "matched": 5, "inherited": 0}
    assert [c["section_title"] for c in chunks] == [title for _p, _t, title in expect]
    assert all(c["section_source"] == "matched" for c in chunks)


def test_chunk_crossing_a_heading_belongs_to_the_section_holding_most_of_it(tmp_path: Path) -> None:
    from tests.pdf_builders import LONG

    st = extract_structure(make_book(tmp_path / "book.pdf"))
    tail_of_intro = "keeps going for several sentences to look like real prose in a document."
    text = f"{tail_of_intro} 1.1 Background " + LONG.format(topic="the history of ice core sampling") * 2
    chunk = _chunk(1, text)
    annotate_chunks(st, [chunk])

    assert chunk["section_title"] == "1.1 Background"          # most of the text
    assert chunk["section_ids"].split(",") == ["s002", "s001"]    # ...but it also touches the intro
    assert st.section_at(chunk["block_idx"]).title == "1 Introduction"   # and it *starts* there


def test_figure_boxes_get_the_section_before_them(tmp_path: Path) -> None:
    st = extract_structure(make_book(tmp_path / "book.pdf"))
    last_goals = max(st.page_blocks(1), key=lambda b: b.bbox[3])
    records = [
        {"figure_id": "a", "page": 1, "bbox": (100, last_goals.bbox[3] + 5, 400, last_goals.bbox[3] + 100)},
        {"figure_id": "b", "page": 2, "bbox": (100, 5, 300, 20)},   # above the first heading of page 2
    ]
    annotate_figures(st, records)

    assert records[0]["section_title"] == "1.2 Goals"
    assert records[1]["section_title"] == "1.2 Goals"   # still the section left open by page 1


def test_two_column_pages_are_read_left_column_first(tmp_path: Path) -> None:
    st = extract_structure(make_two_column(tmp_path / "two.pdf"))
    order = [b.text for b in st.page_blocks(1)]

    left_end = max(i for i, t in enumerate(order) if "left column" in t or "SECONDLEFT" in t)
    right_start = min(i for i, t in enumerate(order) if "RIGHTSECTION" in t or "right column" in t)
    assert left_end < right_start
    assert [t for t, _ in titles(st)] == ["I. LEFTSECTION", "II. SECONDLEFT", "III. RIGHTSECTION"]
    assert st.title == "A Two Column Paper Title"


def test_unreadable_file_returns_none_instead_of_raising(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    assert safe_extract_structure(bad) is None


def test_structure_json_is_serialisable(tmp_path: Path) -> None:
    import json

    payload = extract_structure(make_book(tmp_path / "book.pdf")).to_dict()
    assert payload["section_count"] == 5
    json.dumps(payload)

import shutil
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from backend.config import OcrSettings  # noqa: E402
from backend.ingestion import ocr  # noqa: E402
from backend.ingestion.parsers import parse_pdf  # noqa: E402


def make_scanned_pdf(path: Path, text: str) -> Path:
    """Page 1 has a real text layer; page 2 is a picture of text (a 'scan') with no text layer."""
    scratch = pymupdf.open()
    sp = scratch.new_page(width=612, height=792)
    sp.insert_textbox(pymupdf.Rect(60, 80, 550, 300), text, fontsize=20, fontname="helv")
    pix = sp.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
    scratch.close()

    doc = pymupdf.open()
    digital = doc.new_page(width=612, height=792)
    digital.insert_textbox(pymupdf.Rect(60, 80, 550, 200), "This page has a normal text layer.", fontsize=12, fontname="helv")
    scan = doc.new_page(width=612, height=792)
    scan.insert_image(scan.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()
    return path


needs_tesseract = pytest.mark.skipif(ocr.find_tesseract() is None, reason="Tesseract program not installed")


@needs_tesseract
def test_scanned_page_is_read_and_digital_page_is_not_ocred(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Photosynthesis converts light energy into chemical energy")

    pages = parse_pdf(pdf)

    assert pages[0] == "This page has a normal text layer."          # untouched
    assert pages[1].startswith("[Scanned page - OCR]")
    flat = " ".join(pages[1].lower().split())     # OCR may wrap lines differently
    assert "photosynthesis" in flat and "chemical energy" in flat


def test_missing_tesseract_leaves_scanned_pages_empty_without_error(tmp_path: Path, monkeypatch) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Some words")
    monkeypatch.setattr(ocr, "find_tesseract", lambda configured="": None)
    monkeypatch.setattr(ocr, "_warned", False)

    pages = parse_pdf(pdf)

    assert pages == ["This page has a normal text layer.", ""]


def test_ocr_can_be_switched_off(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Some words")

    assert ocr.ocr_pdf_pages(pdf, [1], OcrSettings(enabled=False)) == {}


def test_unreadable_file_returns_nothing(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")

    assert ocr.ocr_pdf_pages(bad, [0], OcrSettings(tesseract_cmd=shutil.which("tesseract") or "")) == {}


def test_configured_tesseract_path_is_checked(tmp_path: Path) -> None:
    assert ocr.find_tesseract(str(tmp_path / "nope.exe")) is None
    fake = tmp_path / "tesseract.exe"
    fake.write_text("x")
    assert ocr.find_tesseract(str(fake)) == str(fake)


def test_page_without_text_in_a_mocked_reader_still_yields_an_empty_string(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("  Café — 東京  "), FakePage(None)]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))

    assert parse_pdf(tmp_path / "mixed.pdf") == ["Café — 東京", ""]

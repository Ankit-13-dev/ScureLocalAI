from pathlib import Path

from backend.ingestion.parsers import parse_text


def test_parse_text_splits_on_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("First paragraph\n\nSecond paragraph\n\n\nThird", encoding="utf-8")

    assert parse_text(path) == ["First paragraph", "Second paragraph", "Third"]


def test_parse_pdf_preserves_unicode_and_represents_pages_without_text(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    """Image-only pages must not make mixed Unicode PDFs fail or lose page positions."""
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            assert path == str(tmp_path / "mixed.pdf")
            self.pages = [FakePage("  Café — 東京  "), FakePage(None)]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))

    from backend.ingestion.parsers import parse_pdf

    assert parse_pdf(tmp_path / "mixed.pdf") == ["Café — 東京", ""]

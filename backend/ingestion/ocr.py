"""Optional OCR for scanned PDF pages (pages that have no text layer).

PyMuPDF renders the page to an image and Tesseract reads it - fully offline.  Compared with
reading the PDF text layer this is slow (a few seconds per page), so it only runs for pages
where the text layer is empty.

Tesseract is a separate program (not a pip package):
    Windows : install from https://github.com/UB-Mannheim/tesseract/wiki  (default folder is auto-detected)
    macOS   : brew install tesseract
    Linux   : sudo apt install tesseract-ocr

If it is missing, scanned pages simply stay empty, exactly as before - nothing breaks and one
warning is logged.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from backend.config import OcrSettings, get_settings


logger = logging.getLogger(__name__)

_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
_warned = False


def find_tesseract(configured: str = "") -> Optional[str]:
    """Path of the tesseract program, or None."""
    if configured:
        return configured if Path(configured).is_file() else None
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    local = os.environ.get("LOCALAPPDATA")
    candidates = list(_WINDOWS_CANDIDATES)
    if local:
        candidates.append(str(Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe"))
    return next((c for c in candidates if Path(c).is_file()), None)


def _warn_once(message: str, *args: object) -> None:
    global _warned
    if not _warned:
        logger.warning(message, *args)
        _warned = True


def ocr_pdf_pages(pdf_path: Path, page_indices: Iterable[int], settings: Optional[OcrSettings] = None) -> dict[int, str]:
    """OCR the given 0-based pages.  Returns ``{page_index: text}`` (missing / failed pages are omitted)."""
    indices = list(page_indices)
    cfg = settings or get_settings().ocr
    if not indices or not cfg.enabled:
        return {}

    try:
        import pymupdf
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        _warn_once("OCR skipped: %s (install pytesseract and Pillow: pip install -r requirements.txt)", exc)
        return {}

    program = find_tesseract(cfg.tesseract_cmd)
    if program is None:
        _warn_once(
            "OCR skipped: the Tesseract program was not found. Install it (see backend/ingestion/ocr.py) "
            "or set ocr.tesseract_cmd in config.yaml. Scanned pages will be left empty."
        )
        return {}
    pytesseract.pytesseract.tesseract_cmd = program

    results: dict[int, str] = {}
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - unreadable file: leave the pages empty
        logger.warning("OCR could not open %s: %s", pdf_path, exc)
        return {}
    try:
        for index in indices:
            try:
                pix = doc[index].get_pixmap(dpi=cfg.dpi, colorspace=pymupdf.csGRAY, alpha=False)
                image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(image, lang=cfg.language).strip()
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the others
                logger.warning("OCR failed on page %d of %s: %s", index + 1, Path(pdf_path).name, exc)
                continue
            if text:
                results[index] = text
    finally:
        doc.close()
    return results

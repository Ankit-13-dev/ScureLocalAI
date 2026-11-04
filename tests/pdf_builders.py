"""Small synthetic PDFs with known structure, shared by the structure / linking tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import pymupdf

from backend.ingestion.Visualizing import _synthetic_photo

BODY = 10
LONG = (
    "This paragraph explains {topic} in some detail so that it is clearly body text and not a "
    "heading. It keeps going for several sentences to look like real prose in a document. "
)


def _body(page, y0: float, topic: str, repeat: int = 2, x0: float = 60, x1: float = 550, height: float = 90) -> float:
    page.insert_textbox(pymupdf.Rect(x0, y0, x1, y0 + height), (LONG.format(topic=topic)) * repeat, fontsize=BODY, fontname="helv")
    return y0 + height + 6


def _heading(page, y: float, text: str, size: float = 16, x: float = 60) -> float:
    page.insert_text((x, y + size), text, fontsize=size, fontname="hebo")
    return y + size + 10


def _furniture(page, number: int) -> None:
    page.insert_text((60, 28), "Running Title of the Book", fontsize=8, fontname="helv")
    page.insert_text((300, 780), str(number), fontsize=8, fontname="helv")


def make_book(path: Path, outline: bool = False) -> Path:
    """Numbered headings: 1 Introduction (1.1, 1.2), 2 Methods (2.1) over two pages."""
    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    _furniture(p1, 1)
    y = _heading(p1, 60, "1 Introduction", 18)
    y = _body(p1, y, "glaciers and meltwater hydrology")
    y = _heading(p1, y + 4, "1.1 Background", 13)
    y = _body(p1, y, "the history of ice core sampling")
    y = _heading(p1, y + 4, "1.2 Goals", 13)
    y = _body(p1, y, "the research goals and hypotheses")
    p2 = doc.new_page(width=612, height=792)
    _furniture(p2, 2)
    y = _heading(p2, 60, "2 Methods", 18)
    y = _body(p2, y, "the sampling methods and calibration")
    y = _heading(p2, y + 4, "2.1 Data", 13)
    y = _body(p2, y, "the measured temperature dataset")
    if outline:
        doc.set_toc([[1, "1 Introduction", 1], [2, "1.1 Background", 1], [2, "1.2 Goals", 1], [1, "2 Methods", 2], [2, "2.1 Data", 2]])
    doc.save(str(path))
    doc.close()
    return path


def make_plain(path: Path) -> Path:
    """Only body paragraphs: no headings at all."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 60.0
    for topic in ("rivers and deltas", "coastal erosion", "wind patterns"):
        y = _body(page, y, topic)
    doc.save(str(path))
    doc.close()
    return path


def make_two_column(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((200, 60), "A Two Column Paper Title", fontsize=20, fontname="hebo")
    # left column
    y = _heading(page, 100, "I. LEFTSECTION", 12, x=50)
    y = _body(page, y, "the left column topic", 2, 50, 295, 200)
    y = _heading(page, y + 4, "II. SECONDLEFT", 12, x=50)
    _body(page, y, "more left column topic", 2, 50, 295, 200)
    # right column
    y = _heading(page, 100, "III. RIGHTSECTION", 12, x=320)
    _body(page, y, "the right column topic", 2, 320, 565, 200)
    doc.save(str(path))
    doc.close()
    return path


def make_figure_book(path: Path) -> Path:
    """Figure sits in '1 Introduction' but is only *discussed* (cited as Fig. 1) in '2 Results'.

    Nearest-text linking gets this wrong; reference-based linking gets it right.
    """
    jpg = cv2.imencode(".jpg", cv2.cvtColor(_synthetic_photo(), cv2.COLOR_RGB2BGR))[1].tobytes()
    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    _furniture(p1, 1)
    y = _heading(p1, 60, "1 Introduction", 18)
    y = _body(p1, y, "the general motivation of the study")
    p1.insert_image(pymupdf.Rect(150, y + 4, 150 + 300, y + 4 + 200), stream=jpg)
    p1.insert_text((150, y + 4 + 200 + 14), "Fig. 1. Glacier retreat photograph.", fontsize=9, fontname="helv")
    p2 = doc.new_page(width=612, height=792)
    _furniture(p2, 2)
    y = _heading(p2, 60, "2 Results", 18)
    p2.insert_textbox(
        pymupdf.Rect(60, y, 550, y + 120),
        "As shown in Fig. 1, the glacier front retreated markedly over the observation period. "
        "The retreat described in Fig. 1 is consistent with the temperature record. " * 2,
        fontsize=BODY, fontname="helv",
    )
    doc.save(str(path))
    doc.close()
    return path

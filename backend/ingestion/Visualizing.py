"""Visualizing.py - find every *visual* element on every page of a PDF.

Goal of this module (stage 1 of the multimodal upgrade)
-------------------------------------------------------
Before a vision-language model (VLM) can *interpret* an image, the system must
know **which part of a page is an image at all**.  "Image" here is deliberately
broad: photographs of people / animals / birds / scenery, bar / pie / line /
scatter charts, diagrams, flow-charts, maps, illustrations - any pictorial
representation - while body text, equations, tables and ruled boxes are
*not* images.

This module does that localisation.  It does **no** interpretation and loads
**no** neural network, so it is fast, deterministic, fully offline and
CPU-only.  Its output is a list of ``FigureRegion`` objects (page number +
bounding box + source + coarse kind + confidence + caption) which later stages
crop, describe with a local VLM (Ollama / Qwen-VL), embed and store in ChromaDB.

Detection strategy - four complementary signals, merged
-------------------------------------------------------
1. EMBEDDED RASTER IMAGES  (photos, screenshots, charts exported as PNG/JPEG)
   Read straight from the PDF object tree with PyMuPDF ``get_image_info``.
   Exact bounding boxes, zero guessing.  Icons, hairlines, page backgrounds and
   logos repeated on many pages are filtered out.

2. VECTOR GRAPHICS  (charts / diagrams / flow-charts drawn with paths)
   Charts exported from Matplotlib, Excel, PowerPoint, TikZ, draw.io ... are not
   images at all - they are thousands of lines, curves and rectangles.  We read
   every drawing operator with ``get_drawings``, group nearby ones into clusters
   (spatial-hash union-find), then decide per cluster whether it is a figure, a
   table, a boxed paragraph, a rule, or *outlined text* using geometric
   features (curves, diagonals, filled areas, distinct fills, text density).
   Axis labels / legend text sitting next to a cluster are absorbed into it.

3. RASTER LAYOUT ANALYSIS  (scanned pages, or pages with absurdly many paths)
   If a page is one big picture (a scan) the PDF structure tells us nothing, so
   we render the page and analyse pixels with OpenCV: remove ruling lines
   (table / underline suppression), keep large non-text connected components
   (bars, curves, blobs), add a continuous-tone map (photographs), cluster and
   filter.  Text-layer words (if any) are masked out first.

4. MERGE + ENRICH
   Overlapping candidates from different signals are fused ("hybrid"), tiny /
   thin / near-page-sized leftovers are dropped, captions ("Fig. 3 ...") are
   attached, every region is rendered and given a coarse *kind*
   (photo | chart | diagram_or_illustration | unknown) from colour statistics,
   axis / bar / pie cues.  The kind is a routing hint only; the VLM stage makes
   the real decision.

Usage
-----
    python backend/ingestion/Visualizing.py                 # every PDF in data/documents
    python backend/ingestion/Visualizing.py paper.pdf       # one file
    python backend/ingestion/Visualizing.py paper.pdf --pages 2-5 --out out_dir
    python backend/ingestion/Visualizing.py --selftest      # built-in verification

As a library:
    from backend.ingestion.Visualizing import detect_figures
    result = detect_figures("paper.pdf")
    for fig in result.figures:
        print(fig.page, fig.bbox, fig.kind, fig.caption)

Tunables live in ``config.yaml`` under the ``vision:`` key (all optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

try:  # PyMuPDF >= 1.24 exposes the ``pymupdf`` name
    import pymupdf
except ImportError:  # pragma: no cover - very old installs only
    import fitz as pymupdf  # type: ignore[no-redef]

__all__ = [
    "VisionConfig",
    "FigureRegion",
    "PageResult",
    "DocumentResult",
    "load_vision_config",
    "detect_figures",
    "detect_figures_in_page",
    "render_region",
    "build_selftest_pdf",
    "run_selftest",
]

log = logging.getLogger("scurelocalai.visualizing")

ROOT_DIR = Path(__file__).resolve().parents[2]  # backend/ingestion/Visualizing.py -> project root
CONFIG_PATH = ROOT_DIR / "config.yaml"

Box = tuple  # (x0, y0, x1, y1) in PDF points, origin top-left


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class VisionConfig:
    """All tunables.  Override any of them in config.yaml under ``vision:``."""

    # --- rendering / output -------------------------------------------------
    analysis_dpi: int = 96            # dpi used to render crops for kind classification
    raster_dpi: int = 100             # dpi used by the pixel-based layout analysis
    crop_dpi: int = 150               # dpi of saved figure crops (what a VLM will later see)
    annotate_dpi: int = 90            # dpi of annotated page previews
    output_dir: Path = ROOT_DIR / "data" / "figures"

    # --- generic size filters (PDF points; 72 pt = 1 inch) ------------------
    min_side_pt: float = 28.0         # smaller than this on either side = icon / bullet
    min_area_pt2: float = 2400.0      # ~ 49 x 49 pt
    max_aspect: float = 14.0          # more elongated than this = rule / separator
    background_cover: float = 0.85    # region covering this share of the page = background / frame
    scan_cover: float = 0.80          # embedded image covering this share = scanned page

    # --- embedded images ----------------------------------------------------
    drop_recurring: bool = True       # drop logos / letterheads repeated on many pages
    recurring_min_pages: int = 3
    recurring_max_area_ratio: float = 0.12   # ...only when small relative to the page

    # --- vector graphics ----------------------------------------------------
    vector_enabled: bool = True
    cluster_gap_pt: float = 9.0       # drawings closer than this belong to one figure
    vector_min_items: int = 5         # fewer path items than this = box / underline
    vector_min_score: float = 0.45
    table_text_ratio: float = 0.10    # word-area / region-area above this + only orthogonal strokes = table
    max_drawings: int = 25000         # more than this -> fall back to pixel analysis
    label_margin_pt: float = 18.0     # absorb short text (axis ticks, legends) this close to a figure
    max_label_words: int = 14
    label_passes: int = 3

    # --- raster layout analysis --------------------------------------------
    raster_enabled: bool = True
    raster_photo_cells: bool = True   # also detect continuous-tone (photo) areas

    # --- captions -----------------------------------------------------------
    caption_search_pt: float = 54.0
    caption_max_chars: int = 400

    # --- classification -----------------------------------------------------
    classify_kind: bool = True


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, Path):
        p = Path(str(value))
        return p if p.is_absolute() else ROOT_DIR / p
    return value


def load_vision_config(path: Optional[Path] = None, **overrides: Any) -> VisionConfig:
    """Build a VisionConfig from the optional ``vision:`` block of config.yaml.

    Unknown keys are ignored with a warning; missing keys keep their defaults, so the
    module works even if the YAML file has no ``vision`` section at all.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            import yaml  # already a project dependency (config.yaml loader)

            with cfg_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            block = raw.get("vision") or {}
            if isinstance(block, dict):
                data = dict(block)
        except Exception as exc:  # noqa: BLE001 - config problems must never kill detection
            log.warning("Could not read %s (%s); using built-in vision defaults", cfg_path, exc)
    data.update({k: v for k, v in overrides.items() if v is not None})

    defaults = VisionConfig()
    known = {f.name for f in fields(VisionConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            log.warning("Ignoring unknown vision config key: %s", key)
            continue
        kwargs[key] = _coerce(value, getattr(defaults, key))
    return VisionConfig(**kwargs)


# =============================================================================
# Result types
# =============================================================================
@dataclass
class FigureRegion:
    """One detected visual element."""

    figure_id: str                       # e.g. "p004_f01"
    page: int                            # 1-based
    bbox: Box                            # PDF points (x0, y0, x1, y1), origin top-left, unrotated page
    bbox_norm: Box                       # same box as fractions of the page (0..1) - handy for the UI
    source: str                          # embedded_image | vector_graphics | raster_layout | hybrid
    kind: str = "unknown"                # photo | chart | diagram_or_illustration | unknown (coarse hint)
    kind_confidence: float = 0.0
    confidence: float = 0.0              # confidence that this region really is a visual element
    caption: Optional[str] = None
    caption_bbox: Optional[Box] = None
    details: dict = field(default_factory=dict)
    crop_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [round(v, 2) for v in self.bbox]
        d["bbox_norm"] = [round(v, 4) for v in self.bbox_norm]
        if self.caption_bbox:
            d["caption_bbox"] = [round(v, 2) for v in self.caption_bbox]
        d["confidence"] = round(self.confidence, 3)
        d["kind_confidence"] = round(self.kind_confidence, 3)
        return d


@dataclass
class PageResult:
    page: int
    width_pt: float
    height_pt: float
    rotation: int = 0                    # original /Rotate; analysis is done in unrotated space
    is_scanned: bool = False
    figures: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # candidates rejected + why (debugging / tuning)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "width_pt": round(self.width_pt, 2),
            "height_pt": round(self.height_pt, 2),
            "rotation": self.rotation,
            "is_scanned": self.is_scanned,
            "figures": [f.to_dict() for f in self.figures],
            "skipped": self.skipped,
            "notes": self.notes,
        }


@dataclass
class DocumentResult:
    path: str
    page_count: int
    pages: list = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def figures(self) -> list:
        return [f for p in self.pages for f in p.figures]

    def to_dict(self) -> dict:
        figs = self.figures
        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for f in figs:
            by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
            by_source[f.source] = by_source.get(f.source, 0) + 1
        return {
            "path": self.path,
            "page_count": self.page_count,
            "figure_count": len(figs),
            "by_kind": by_kind,
            "by_source": by_source,
            "elapsed_s": round(self.elapsed_s, 2),
            "pages": [p.to_dict() for p in self.pages],
        }


# =============================================================================
# Geometry helpers (boxes are plain tuples: x0, y0, x1, y1)
# =============================================================================
def _w(b: Box) -> float:
    return b[2] - b[0]


def _h(b: Box) -> float:
    return b[3] - b[1]


def _area(b: Box) -> float:
    return max(0.0, _w(b)) * max(0.0, _h(b))


def _inter_area(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _expand(b: Box, m: float) -> Box:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _clip(b: Box, page_box: Box) -> Box:
    return (max(b[0], page_box[0]), max(b[1], page_box[1]), min(b[2], page_box[2]), min(b[3], page_box[3]))


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _iou(a: Box, b: Box) -> float:
    inter = _inter_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _center(b: Box) -> tuple:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _center_inside(inner: Box, outer: Box) -> bool:
    cx, cy = _center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _cluster_boxes(boxes: Sequence[Box], gap: float) -> list:
    """Union-find clustering: boxes whose gap is <= ``gap`` share a cluster.

    A spatial hash keeps this near-linear even for pages with 10k+ path items.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    cell = max(gap * 4.0, 32.0)
    grid: dict = {}
    grown = [_expand(b, gap / 2.0) for b in boxes]
    for i, g in enumerate(grown):
        for cx in range(int(g[0] // cell), int(g[2] // cell) + 1):
            for cy in range(int(g[1] // cell), int(g[3] // cell) + 1):
                bucket = grid.setdefault((cx, cy), [])
                for j in bucket:
                    if _overlaps(g, grown[j]):
                        union(i, j)
                bucket.append(i)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# =============================================================================
# PDF access helpers
# =============================================================================
def _render(page: "pymupdf.Page", clip: Optional[Box], dpi: int) -> np.ndarray:
    """Render a page (or a clip of it) to an RGB uint8 array."""
    rect = pymupdf.Rect(*clip) if clip else None
    pix = page.get_pixmap(dpi=dpi, clip=rect, colorspace=pymupdf.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].copy()


def render_region(page: "pymupdf.Page", bbox: Box, dpi: int = 150) -> np.ndarray:
    """Public helper: crop ``bbox`` (PDF points) from ``page`` at ``dpi`` as an RGB array."""
    return _render(page, bbox, dpi)


def _encode_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _write_png(path: Path, rgb: np.ndarray) -> None:
    # cv2.imwrite breaks on non-ASCII Windows paths; encoding in memory + Path.write_bytes does not.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encode_png(rgb))


@dataclass
class _TextBlock:
    bbox: Box
    n_words: int
    text: str


def _text_blocks(page: "pymupdf.Page") -> list:
    out = []
    for b in page.get_text("blocks"):
        if len(b) >= 7 and b[6] != 0:  # 0 = text block, 1 = image block
            continue
        text = " ".join(str(b[4]).split())
        if not text:
            continue
        out.append(_TextBlock((float(b[0]), float(b[1]), float(b[2]), float(b[3])), len(text.split()), text))
    return out


def _words(page: "pymupdf.Page") -> list:
    return [(float(w[0]), float(w[1]), float(w[2]), float(w[3])) for w in page.get_text("words")]


def _text_lines(page: "pymupdf.Page") -> list:
    """Individual text lines (finer than blocks, so unrelated labels never fuse into one box)."""
    groups: dict = {}
    for w in page.get_text("words"):
        key = (w[5], w[6])  # block_no, line_no
        g = groups.setdefault(key, [float(w[0]), float(w[1]), float(w[2]), float(w[3]), []])
        g[0], g[1], g[2], g[3] = min(g[0], w[0]), min(g[1], w[1]), max(g[2], w[2]), max(g[3], w[3])
        g[4].append(str(w[4]))
    return [_TextBlock((g[0], g[1], g[2], g[3]), len(g[4]), " ".join(g[4])) for g in groups.values()]


def _is_white(color: Any) -> bool:
    return color is not None and len(color) >= 1 and all(c >= 0.97 for c in color)


# =============================================================================
# Signal 1 - embedded raster images
# =============================================================================
def _collect_image_infos(page: "pymupdf.Page") -> list:
    try:
        return page.get_image_info(xrefs=True)
    except Exception as exc:  # noqa: BLE001 - a broken image object must not abort the page
        log.warning("get_image_info failed on page %s: %s", page.number + 1, exc)
        return []


def _recurring_digests(all_infos: Sequence[list], page_areas: Sequence[float], cfg: VisionConfig) -> set:
    """Digests of small images that repeat on many pages (logos, letterheads, watermarks)."""
    seen: dict = {}
    for infos, parea in zip(all_infos, page_areas):
        pages_digests = set()
        for info in infos:
            digest = info.get("digest")
            if not digest:
                continue
            b = tuple(info["bbox"])
            if parea > 0 and _area(b) / parea <= cfg.recurring_max_area_ratio:
                pages_digests.add(digest)
        for d in pages_digests:
            seen[d] = seen.get(d, 0) + 1
    return {d for d, c in seen.items() if c >= cfg.recurring_min_pages}


def _embedded_candidates(infos: list, page_box: Box, cfg: VisionConfig, recurring: set, skipped: list) -> tuple:
    """Return (candidates, full_page_image_found)."""
    page_area = _area(page_box)
    cands: list = []
    full_page = False
    for info in infos:
        box = _clip(tuple(float(v) for v in info["bbox"]), page_box)
        if _w(box) <= 0 or _h(box) <= 0:
            continue
        cover = _area(box) / page_area
        if cover >= cfg.scan_cover:
            full_page = True
            skipped.append({"bbox": _r(box), "source": "embedded_image", "reason": "full_page_image"})
            continue
        if info.get("digest") in recurring and cfg.drop_recurring:
            skipped.append({"bbox": _r(box), "source": "embedded_image", "reason": "recurring_decoration"})
            continue
        reason = _size_reject(box, cfg)
        if reason:
            skipped.append({"bbox": _r(box), "source": "embedded_image", "reason": reason})
            continue
        cands.append(
            {
                "box": box,
                "source": "embedded_image",
                "score": 0.97,
                "details": {
                    "xref": info.get("xref"),
                    "native_px": [info.get("width"), info.get("height")],
                    "colorspace": info.get("cs-name"),
                },
            }
        )
    return cands, full_page


def _size_reject(box: Box, cfg: VisionConfig) -> Optional[str]:
    w, h = _w(box), _h(box)
    if min(w, h) < cfg.min_side_pt:
        return "too_small_or_thin"
    if w * h < cfg.min_area_pt2:
        return "too_small"
    if max(w, h) / max(1e-6, min(w, h)) > cfg.max_aspect:
        return "rule_like_aspect"
    return None


def _r(box: Box) -> list:
    return [round(float(v), 1) for v in box]


# =============================================================================
# Signal 2 - vector graphics
# =============================================================================
@dataclass
class _Drawing:
    rect: Box
    n_lines_orth: int = 0
    n_lines_diag: int = 0
    n_curves: int = 0
    n_rects: int = 0
    filled: bool = False                 # painted with a non-white fill
    fill: Optional[tuple] = None


def _parse_drawings(page: "pymupdf.Page", page_box: Box, cfg: VisionConfig) -> Optional[list]:
    try:
        raw = page.get_drawings()
    except Exception as exc:  # noqa: BLE001
        log.warning("get_drawings failed on page %s: %s", page.number + 1, exc)
        return []
    if len(raw) > cfg.max_drawings:
        return None  # caller falls back to pixel analysis
    page_area = _area(page_box)
    out: list = []
    for d in raw:
        r = d.get("rect")
        if r is None:
            continue
        rect = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        if _w(rect) < 0.01 and _h(rect) < 0.01:
            continue
        if _area(rect) >= cfg.background_cover * page_area:
            continue  # full-page background fill / page border frame
        dr = _Drawing(rect=rect)
        fill = d.get("fill")
        if fill is not None and not _is_white(fill):
            dr.filled = True
            dr.fill = tuple(round(float(c), 2) for c in fill)
        pts: list = []
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                pts += [(item[1].x, item[1].y), (item[2].x, item[2].y)]
            elif kind == "c":
                pts += [(q.x, q.y) for q in item[1:5]]
            elif kind in ("re", "qu"):
                rr = item[1].rect if kind == "qu" else item[1]
                pts += [(rr.x0, rr.y0), (rr.x1, rr.y1)]
        if pts:  # PyMuPDF's own d["rect"] can under-report multi-segment open paths; recompute
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            dr.rect = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                if dx < 0.8 or dy < 0.8:
                    dr.n_lines_orth += 1
                else:
                    dr.n_lines_diag += 1
            elif kind == "c":
                dr.n_curves += 1
            elif kind in ("re", "qu"):
                dr.n_rects += 1
        out.append(dr)
    return out


def _vector_candidates(
    drawings: list,
    words: list,
    blocks: list,  # text *lines*, used to absorb axis / legend labels
    page_box: Box,
    cfg: VisionConfig,
    skipped: list,
) -> list:
    if not drawings:
        return []
    boxes = [d.rect for d in drawings]
    cands: list = []
    for members in _cluster_boxes(boxes, cfg.cluster_gap_pt):
        ds = [drawings[i] for i in members]
        box = boxes[members[0]]
        for i in members[1:]:
            box = _union(box, boxes[i])
        box = _clip(box, page_box)
        feats = _cluster_features(ds, box, words)
        decision, score, why = _judge_vector_cluster(feats, box, cfg)
        if decision != "figure":
            if _area(box) >= 400:  # don't flood the debug list with specks
                skipped.append({"bbox": _r(box), "source": "vector_graphics", "reason": why, "items": feats["items"]})
            continue
        box = _absorb_labels(box, blocks, page_box, cfg)
        reason = _size_reject(box, cfg)
        if reason:
            skipped.append({"bbox": _r(box), "source": "vector_graphics", "reason": reason})
            continue
        cands.append({"box": box, "source": "vector_graphics", "score": score, "details": _public_vector_feats(feats)})
    return cands


def _cluster_features(ds: list, box: Box, words: list) -> dict:
    items = sum(d.n_lines_orth + d.n_lines_diag + d.n_curves + d.n_rects for d in ds)
    fills = {d.fill for d in ds if d.filled}
    filled_rects = sum(1 for d in ds if d.filled and d.n_rects > 0)
    barea = max(_area(box), 1.0)
    inside = [w for w in words if _center_inside(w, box)]
    word_area = sum(_area(w) for w in inside)

    # "Outlined text" detector: PDFs whose glyphs were converted to paths produce a
    # cluster of many small solid shapes of many *different* sizes.  Scatter-plot
    # markers are also small and solid but come in only a few sizes.
    small_solid = [d for d in ds if d.filled and _w(d.rect) < 26 and _h(d.rect) < 26]
    sizes = {(round(_w(d.rect)), round(_h(d.rect))) for d in small_solid}

    return {
        "items": items,
        "n_drawings": len(ds),
        "n_curves": sum(d.n_curves for d in ds),
        "n_diag": sum(d.n_lines_diag for d in ds),
        "n_orth": sum(d.n_lines_orth for d in ds),
        "n_rects": sum(d.n_rects for d in ds),
        "filled_rects": filled_rects,
        "distinct_fills": len(fills),
        "n_words": len(inside),
        "text_ratio": word_area / barea,
        "small_solid": len(small_solid),
        "small_solid_sizes": len(sizes),
    }


def _public_vector_feats(f: dict) -> dict:
    keep = ("items", "n_curves", "n_diag", "n_orth", "n_rects", "filled_rects", "distinct_fills", "n_words")
    out = {k: f[k] for k in keep}
    out["text_ratio"] = round(f["text_ratio"], 3)
    return out


def _judge_vector_cluster(f: dict, box: Box, cfg: VisionConfig) -> tuple:
    """Return (decision, score, reason) for one cluster of path items."""
    if f["n_drawings"] >= 30 and f["small_solid"] / f["n_drawings"] >= 0.8 and f["small_solid_sizes"] >= 8:
        return "reject", 0.0, "outlined_text"
    if f["items"] < cfg.vector_min_items:
        return "reject", 0.0, "too_few_items"

    curvy = f["n_curves"] >= 1 or f["n_diag"] >= 3
    words_per_item = f["n_words"] / max(1, f["items"])
    text_dense = f["text_ratio"] >= cfg.table_text_ratio or words_per_item >= 1.5

    if not curvy:
        # Only horizontal / vertical strokes and rectangles: table, grid, boxed text or a bar chart.
        if f["filled_rects"] >= 3 and not text_dense:
            return "figure", 0.72, "bars"
        if text_dense:
            return "reject", 0.0, "table_or_boxed_text"
        if f["items"] >= 12:
            return "figure", 0.50, "orthogonal_graphic"
        return "reject", 0.0, "ruled_lines"

    if f["n_curves"] < 8 and f["n_diag"] < 3 and text_dense:
        return "reject", 0.0, "rounded_text_box"

    score = 0.50 + min(0.30, (f["n_curves"] + f["n_diag"]) / 60.0)
    if f["distinct_fills"] >= 3:
        score += 0.10
    if f["text_ratio"] > 0.35:
        score -= 0.30
    score = max(0.0, min(0.95, score))
    if score < cfg.vector_min_score:
        return "reject", score, "low_vector_score"
    return "figure", score, "curves_or_diagonals"


CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?s?|chart|graph|diagram|plate|photo(?:graph)?|image|illustration|scheme|map|exhibit)\b\.?\s*"
    r"[\dIVXivx]*[.:)\-\u2013\u2014]?",
    re.IGNORECASE,
)


def _absorb_labels(box: Box, blocks: list, page_box: Box, cfg: VisionConfig) -> Box:
    """Grow a vector figure to include its axis ticks, axis titles and legend text."""
    cur = box
    for _ in range(cfg.label_passes):
        grown = _expand(cur, cfg.label_margin_pt)
        changed = False
        for blk in blocks:
            if blk.n_words > cfg.max_label_words or CAPTION_RE.match(blk.text):
                continue
            if _overlaps(grown, blk.bbox) and _inter_area(grown, blk.bbox) >= 0.5 * _area(blk.bbox):
                new = _union(cur, blk.bbox)
                if new != cur:
                    cur, changed = new, True
        if not changed:
            break
    return _clip(cur, page_box)


# =============================================================================
# Signal 3 - raster layout analysis (scans / path-heavy pages)
# =============================================================================
def _raster_candidates(
    page: "pymupdf.Page", page_box: Box, words: list, cfg: VisionConfig, skipped: list
) -> list:
    dpi = cfg.raster_dpi
    scale = dpi / 72.0
    img = _render(page, None, dpi)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Words from a text layer (if the scan was OCR'd) are known text: blank them out.
    text_mask = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in words:
        a, b = max(0, int(x0 * scale) - 1), max(0, int(y0 * scale) - 1)
        c, d = min(W, int(x1 * scale) + 2), min(H, int(y1 * scale) + 2)
        text_mask[b:d, a:c] = 1

    paper = float(np.clip(np.percentile(gray, 90), 150, 255))
    ink = ((gray < paper - 45) | (sat > 55)) & (text_mask == 0)
    ink_u8 = ink.astype(np.uint8)

    # --- 1. strip ruling lines (table borders, underlines, frames, axes) --------------------
    L = max(15, int(0.55 * dpi))
    thin = max(3, int(0.03 * dpi))
    thick_k = max(7, int(0.07 * dpi))

    def thin_lines(kernel_len: int, horizontal: bool) -> np.ndarray:
        k_thin = (kernel_len, 1) if horizontal else (1, kernel_len)
        k_thick = (kernel_len, thick_k) if horizontal else (thick_k, kernel_len)
        any_run = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_thin))
        thick = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_thick))
        thick = cv2.dilate(thick, np.ones((3, 3), np.uint8))
        return (any_run > 0) & (thick == 0)

    lines = thin_lines(L, True) | thin_lines(L, False)
    lines_dil = cv2.dilate(lines.astype(np.uint8), np.ones((thin, thin), np.uint8)) > 0
    rest = (ink & ~lines_dil).astype(np.uint8)

    # --- 2. graphic seeds: big components that are not text-sized --------------------------
    n, _, st, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    text_max = 0.45 * dpi
    min_side = 0.25 * dpi
    seed_mask = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) > text_max and min(w, h) > min_side and a > 0.004 * dpi * dpi:
            seed_mask[y : y + h, x : x + w] = 1

    boxes_px: list = []  # (x0, y0, x1, y1, cue)
    if seed_mask.any():
        merged = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.25 * dpi), int(0.25 * dpi))))
        nn, _, st2, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        for i in range(1, nn):
            x, y, w, h, _a = st2[i]
            pad = int(0.25 * dpi) // 2
            boxes_px.append([x + pad, y + pad, x + w - pad, y + h - pad, "structure"])

    # --- 3. continuous-tone cells (photographs) --------------------------------------------
    if cfg.raster_photo_cells:
        boxes_px.extend(_photo_boxes(gray, sat, paper, text_mask, dpi))

    if not boxes_px:
        return []

    # --- 4. fuse overlapping boxes, absorb labels / axis lines / ticks ----------------------
    boxes_px = _merge_px_boxes(boxes_px)
    ncc, _, stc, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    comp_boxes = [(stc[i][0], stc[i][1], stc[i][0] + stc[i][2], stc[i][1] + stc[i][3]) for i in range(1, ncc)]
    margin = int(0.28 * dpi)
    out: list = []
    for b in boxes_px:
        x0, y0, x1, y1, cue = b
        for _ in range(3):
            grown = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
            changed = False
            for cb in comp_boxes:
                small = max(cb[2] - cb[0], cb[3] - cb[1]) <= text_max or (cb[2] - cb[0] > L) != (cb[3] - cb[1] > L)
                if small and _overlaps(grown, cb) and not _contains_px((x0, y0, x1, y1), cb):
                    nx0, ny0, nx1, ny1 = min(x0, cb[0]), min(y0, cb[1]), max(x1, cb[2]), max(y1, cb[3])
                    if (nx1 - nx0) * (ny1 - ny0) <= 1.6 * (x1 - x0) * (y1 - y0) + 2500:
                        x0, y0, x1, y1, changed = nx0, ny0, nx1, ny1, True
            if not changed:
                break
        box = _clip((x0 / scale, y0 / scale, x1 / scale, y1 / scale), page_box)
        if _area(box) >= cfg.background_cover * _area(page_box):
            skipped.append({"bbox": _r(box), "source": "raster_layout", "reason": "page_background"})
            continue
        reason = _size_reject(box, cfg)
        if reason:
            skipped.append({"bbox": _r(box), "source": "raster_layout", "reason": reason})
            continue
        words_in = sum(1 for w in words if _center_inside(w, box))
        if words_in and sum(_area(w) for w in words if _center_inside(w, box)) / _area(box) > 0.30 and cue != "tone":
            skipped.append({"bbox": _r(box), "source": "raster_layout", "reason": "text_dominated"})
            continue
        out.append(
            {
                "box": box,
                "source": "raster_layout",
                "score": 0.80 if cue == "tone" else 0.65,
                "details": {"cue": "continuous_tone" if cue == "tone" else "structure"},
            }
        )
    return out


def _contains_px(outer: tuple, inner: tuple) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _merge_px_boxes(boxes: list) -> list:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list = []
        for b in boxes:
            for o in out:
                if _overlaps(tuple(b[:4]), tuple(o[:4])):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    o[4] = "tone" if "tone" in (o[4], b[4]) else o[4]
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _photo_boxes(gray: np.ndarray, sat: np.ndarray, paper: float, text_mask: np.ndarray, dpi: int) -> list:
    """Find large continuous-tone areas: plenty of mid-tones that are *textured*, not flat fills."""
    H, W = gray.shape
    cell = max(8, int(0.14 * dpi))
    gh, gw = H // cell, W // cell
    if gh < 3 or gw < 3:
        return []
    g = gray[: gh * cell, : gw * cell].astype(np.float32)
    s = sat[: gh * cell, : gw * cell].astype(np.float32)
    tm = text_mask[: gh * cell, : gw * cell].astype(np.float32)

    mid = ((g > 30) & (g < paper - 30)).astype(np.float32)
    colourful = (s > 45).astype(np.float32)
    tone = np.maximum(mid, colourful)

    def cell_mean(a: np.ndarray) -> np.ndarray:
        return cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)

    tone_c = cell_mean(tone)
    mean = cell_mean(g)
    var = np.maximum(cell_mean(g * g) - mean * mean, 0.0)
    textured = np.sqrt(var) > 6.0
    known_text = cell_mean(tm) > 0.3

    photo_cells = ((tone_c > 0.72) & textured & ~known_text).astype(np.uint8)
    photo_cells = cv2.morphologyEx(photo_cells, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    photo_cells = cv2.morphologyEx(photo_cells, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(photo_cells, connectivity=8)
    out: list = []
    min_cells = int(0.8 * dpi / cell) ** 2  # at least ~0.8 x 0.8 inch of continuous tone
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_cells and w >= 3 and h >= 3 and a / float(w * h) > 0.55:
            out.append([x * cell, y * cell, (x + w) * cell, (y + h) * cell, "tone"])
    return out


# =============================================================================
# Merging, captions, classification
# =============================================================================
def _merge_candidates(cands: list) -> list:
    """Fuse candidates that *overlap* (a chart drawn over an image, a frame around a photo...).

    Candidates that merely sit next to each other (two side-by-side plots) stay separate.
    """
    items = [dict(c, sources={c["source"]}) for c in cands]
    changed = True
    while changed:
        changed = False
        out: list = []
        for c in items:
            for o in out:
                inter = _inter_area(c["box"], o["box"])
                smaller = min(_area(c["box"]), _area(o["box"]))
                if smaller > 0 and inter / smaller >= 0.15:
                    o["box"] = _union(o["box"], c["box"])
                    o["score"] = max(o["score"], c["score"])
                    o["sources"] |= c["sources"]
                    o["details"] = {**c["details"], **o["details"]}
                    changed = True
                    break
            else:
                out.append(c)
        items = out
    for c in items:
        c["source"] = c["source"] if len(c["sources"]) == 1 else "hybrid"
    return items


def _find_caption(box: Box, blocks: list, cfg: VisionConfig) -> tuple:
    best = None
    for blk in blocks:
        if not CAPTION_RE.match(blk.text):
            continue
        bb = blk.bbox
        overlap_x = min(box[2], bb[2]) - max(box[0], bb[0])
        if overlap_x < 0.3 * min(_w(box), _w(bb)):
            continue
        below = bb[1] - box[3]
        above = box[1] - bb[3]
        if -3 <= below <= cfg.caption_search_pt:
            dist, side = max(below, 0.0), "below"
        elif -3 <= above <= cfg.caption_search_pt:
            dist, side = max(above, 0.0) + 6.0, "above"  # prefer captions below
        else:
            continue
        if best is None or dist < best[0]:
            best = (dist, blk, side)
    if best is None:
        return None, None, None
    text = best[1].text
    if len(text) > cfg.caption_max_chars:
        text = text[: cfg.caption_max_chars].rstrip() + "..."
    return text, best[1].bbox, best[2]


def _classify_crop(rgb: np.ndarray) -> tuple:
    """Coarse kind from pixel statistics.  Returns (kind, confidence, cues)."""
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return "unknown", 0.0, {}
    sc = 360.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]

    # -- colour statistics: photos are spread over many colours / grey levels, charts over a handful
    q = (small >> 3).astype(np.uint32)                    # 5 bits per channel
    key = (q[:, :, 0] << 10) | (q[:, :, 1] << 5) | q[:, :, 2]
    counts = np.bincount(key.ravel(), minlength=32768).astype(np.float64)
    total = counts.sum()
    top8 = float(np.sort(counts)[::-1][:8].sum() / total)
    n_colors = int((counts > 0.002 * total).sum())
    hist = np.bincount((gray >> 2).ravel(), minlength=64).astype(np.float64) / gray.size
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())   # 0..6 bits; photos high, charts low
    mid_frac = float(((gray > 30) & (gray < 225)).mean())

    # -- local texture: photos have smooth gradients / noise everywhere
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
    nonwhite = (gray < 235) | (sat > 30)
    textured = float(((np.abs(lap) > 1.5) & nonwhite).sum() / max(1, nonwhite.sum()))

    photo = (
        0.30 * np.clip((0.80 - top8) / 0.40, 0, 1)
        + 0.30 * np.clip((entropy - 2.5) / 2.5, 0, 1)
        + 0.15 * np.clip((n_colors - 20) / 80.0, 0, 1)
        + 0.10 * np.clip((mid_frac - 0.25) / 0.40, 0, 1)
        + 0.15 * np.clip((textured - 0.30) / 0.40, 0, 1)
    )

    # -- chart cues: axes, solid bars, circular (pie) bodies
    mask = ((gray < 205) | (sat > 40)).astype(np.uint8)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(0.4 * sw)), 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(0.4 * sh))))
    has_h = bool(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_h).any())
    has_v = bool(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_v).any())
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    solid = 0
    pie = False
    for i in range(1, n):
        x, y, cw, ch, a = st[i]
        if a < 25:
            continue
        fill = a / float(cw * ch)
        if 6 <= cw and 6 <= ch and fill > 0.88 and cw * ch < 0.4 * sw * sh:
            solid += 1
        if a > 0.15 * sw * sh and 0.85 <= cw / float(ch) <= 1.18 and fill > 0.68 and a / (np.pi * (max(cw, ch) / 2.0) ** 2) > 0.8:
            pie = True
    chart = 0.0
    chart += 0.45 if (has_h and has_v) else (0.15 if (has_h or has_v) else 0.0)
    chart += 0.35 if solid >= 3 else 0.0
    chart += 0.40 if pie else 0.0
    chart = float(min(1.0, chart))

    cues = {
        "top8_color_share": round(top8, 3),
        "gray_entropy_bits": round(entropy, 2),
        "n_dominant_colors": n_colors,
        "photo_score": round(float(photo), 3),
        "chart_score": round(chart, 3),
        "axes": bool(has_h and has_v),
        "solid_blocks": int(solid),
        "pie_like": bool(pie),
    }
    if photo >= 0.50:  # real charts score <0.2 on photo statistics, so this gap is wide
        return "photo", float(min(1.0, 0.5 + photo / 2.0)), cues
    if chart >= 0.45:
        return "chart", float(min(1.0, 0.5 + chart / 2.0)), cues
    if photo < 0.35:
        return "diagram_or_illustration", float(0.55 + 0.3 * (1 - photo)), cues
    return "unknown", 0.4, cues


# =============================================================================
# Page / document drivers
# =============================================================================
def _normalise_rotation(page: "pymupdf.Page") -> int:
    """Work in unrotated page space so text, drawings and image boxes share a frame."""
    rot = int(page.rotation or 0)
    if rot:
        page.set_rotation(0)  # in-memory only; the file on disk is never modified
    return rot


def detect_figures_in_page(
    page: "pymupdf.Page",
    cfg: Optional[VisionConfig] = None,
    infos: Optional[list] = None,
    recurring: Optional[set] = None,
) -> PageResult:
    """Detect all visual regions on one page."""
    cfg = cfg or VisionConfig()
    rotation = _normalise_rotation(page)
    pr = page.rect
    page_box: Box = (float(pr.x0), float(pr.y0), float(pr.x1), float(pr.y1))
    res = PageResult(page=page.number + 1, width_pt=_w(page_box), height_pt=_h(page_box), rotation=rotation)
    skipped = res.skipped

    words = _words(page)
    blocks = _text_blocks(page)
    lines = _text_lines(page)
    infos = _collect_image_infos(page) if infos is None else infos

    cands, full_page_image = _embedded_candidates(infos, page_box, cfg, recurring or set(), skipped)

    need_raster = False
    if full_page_image:
        res.is_scanned = True
        need_raster = True
        res.notes.append("full-page image detected: page treated as scanned, pixel analysis used")

    if cfg.vector_enabled:
        drawings = _parse_drawings(page, page_box, cfg)
        if drawings is None:
            need_raster = True
            res.notes.append(f"more than {cfg.max_drawings} path items: pixel analysis used instead of vector analysis")
        elif not full_page_image:  # vector paths on a scanned page are just OCR-layer noise
            cands.extend(_vector_candidates(drawings, words, lines, page_box, cfg, skipped))

    if need_raster and cfg.raster_enabled:
        cands.extend(_raster_candidates(page, page_box, words, cfg, skipped))

    merged = _merge_candidates(cands)
    merged = [c for c in merged if _size_reject(c["box"], cfg) is None or c["source"] == "embedded_image"]
    merged.sort(key=lambda c: (round(c["box"][1] / 12.0), c["box"][0]))

    for idx, c in enumerate(merged, start=1):
        box = c["box"]
        caption, cap_box, cap_side = _find_caption(box, blocks, cfg)
        fig = FigureRegion(
            figure_id=f"p{res.page:03d}_f{idx:02d}",
            page=res.page,
            bbox=box,
            bbox_norm=(box[0] / _w(page_box), box[1] / _h(page_box), box[2] / _w(page_box), box[3] / _h(page_box)),
            source=c["source"],
            confidence=float(c["score"]),
            caption=caption,
            caption_bbox=cap_box,
            details=dict(c["details"]),
        )
        if cap_side:
            fig.details["caption_side"] = cap_side
        if cfg.classify_kind:
            try:
                crop = _render(page, box, cfg.analysis_dpi)
                fig.kind, fig.kind_confidence, cues = _classify_crop(crop)
                fig.details["kind_cues"] = cues
            except Exception as exc:  # noqa: BLE001 - classification is advisory only
                log.warning("kind classification failed for %s: %s", fig.figure_id, exc)
        res.figures.append(fig)
    return res


def _parse_pages(spec: Optional[str], count: int) -> list:
    """'1-3,7' -> [0,1,2,6] (0-based, clamped).  None -> all pages."""
    if not spec:
        return list(range(count))
    chosen: list = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a or 1), int(b or count)
        else:
            lo = hi = int(part)
        chosen.extend(range(max(1, lo) - 1, min(count, hi)))
    return sorted(set(chosen))


def detect_figures(
    pdf_path: Any,
    cfg: Optional[VisionConfig] = None,
    pages: Optional[Iterable[int]] = None,
    page_spec: Optional[str] = None,
) -> DocumentResult:
    """Detect visual regions in a whole PDF.

    ``pages`` is an optional iterable of 1-based page numbers; ``page_spec`` accepts "1-3,7".
    """
    cfg = cfg or load_vision_config()
    t0 = time.perf_counter()
    doc = pymupdf.open(str(pdf_path))
    try:
        if doc.needs_pass:
            raise ValueError(f"{pdf_path} is password protected")
        idxs = sorted({p - 1 for p in pages if 1 <= p <= len(doc)}) if pages is not None else _parse_pages(page_spec, len(doc))

        # pass 1: image digests across *all* selected pages, to spot repeated logos
        all_infos: list = []
        areas: list = []
        for i in idxs:
            pg = doc[i]
            _normalise_rotation(pg)
            all_infos.append(_collect_image_infos(pg))
            areas.append(pg.rect.width * pg.rect.height)
        recurring = _recurring_digests(all_infos, areas, cfg)

        result = DocumentResult(path=str(pdf_path), page_count=len(doc))
        for i, infos in zip(idxs, all_infos):
            try:
                result.pages.append(detect_figures_in_page(doc[i], cfg, infos=infos, recurring=recurring))
            except Exception as exc:  # noqa: BLE001 - one bad page must not lose the whole document
                log.exception("page %s failed", i + 1)
                pr = PageResult(page=i + 1, width_pt=0.0, height_pt=0.0)
                pr.notes.append(f"page analysis failed: {exc}")
                result.pages.append(pr)
        result.elapsed_s = time.perf_counter() - t0
        return result
    finally:
        doc.close()


# =============================================================================
# Output: crops, annotated previews, JSON, console summary
# =============================================================================
KIND_COLORS = {  # RGB
    "photo": (220, 50, 50),
    "chart": (40, 110, 230),
    "diagram_or_illustration": (30, 160, 80),
    "unknown": (230, 140, 20),
}


def save_outputs(pdf_path: Path, result: DocumentResult, out_dir: Path, cfg: VisionConfig, images: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if not images:
        return out_dir
    doc = pymupdf.open(str(pdf_path))
    try:
        for pr in result.pages:
            if not pr.figures:
                continue
            page = doc[pr.page - 1]
            _normalise_rotation(page)
            for fig in pr.figures:
                crop_file = out_dir / "crops" / f"{fig.figure_id}.png"
                _write_png(crop_file, _render(page, fig.bbox, cfg.crop_dpi))
                fig.crop_path = str(crop_file)
            # annotated page preview
            preview = _render(page, None, cfg.annotate_dpi)
            sc = cfg.annotate_dpi / 72.0
            bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            for fig in pr.figures:
                x0, y0, x1, y1 = [int(v * sc) for v in fig.bbox]
                col = KIND_COLORS.get(fig.kind, KIND_COLORS["unknown"])[::-1]
                cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 2)
                label = f"{fig.figure_id[-3:]} {fig.kind.split('_')[0]} {fig.confidence:.2f}"
                cv2.putText(bgr, label, (x0 + 3, max(12, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            _write_png(out_dir / f"page_{pr.page:03d}_annotated.png", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        doc.close()
    # crop_path values were filled in -> rewrite JSON so it references the crops
    (out_dir / "figures.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def print_summary(result: DocumentResult, out: Any = None) -> None:
    out = out or sys.stdout
    d = result.to_dict()
    print(f"\n=== {Path(result.path).name}  ({result.page_count} pages, {result.elapsed_s:.2f}s) ===", file=out)
    print(f"figures: {d['figure_count']}   by kind: {d['by_kind']}   by source: {d['by_source']}", file=out)
    for pr in result.pages:
        tag = " [scanned]" if pr.is_scanned else ""
        if not pr.figures:
            if pr.notes:
                print(f"  p{pr.page:>3}{tag}: none  ({'; '.join(pr.notes)})", file=out)
            continue
        print(f"  p{pr.page:>3}{tag}: {len(pr.figures)} figure(s)", file=out)
        for f in pr.figures:
            bx = ", ".join(f"{v:6.1f}" for v in f.bbox)
            cap = f'  caption="{f.caption[:60]}..."' if f.caption else ""
            print(f"       {f.figure_id}  [{bx}]  {f.source:<15} {f.kind:<24} conf={f.confidence:.2f}{cap}", file=out)


# =============================================================================
# Self-test: build a synthetic PDF with known ground truth and check detection
# =============================================================================
def _synthetic_photo(w: int = 720, h: int = 480, seed: int = 7) -> np.ndarray:
    """A landscape-like colour image (sky gradient, sun, hills, noise) - no external files needed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack([90 + 120 * yy / h, 150 + 80 * yy / h, 235 - 60 * yy / h], axis=2)
    sun = np.exp(-(((xx - 0.72 * w) ** 2 + (yy - 0.25 * h) ** 2) / (2 * (0.07 * w) ** 2)))[..., None]
    sky = sky + sun * np.array([120, 100, 20], np.float32)
    ridge = 0.62 * h + 28 * np.sin(xx / 55.0) + 14 * np.sin(xx / 17.0 + 1.3)
    land = (yy > ridge)[..., None]
    grass = np.stack([40 + 50 * np.sin(xx / 9.0) ** 2, 110 + 60 * yy / h, 45 + 30 * np.cos(yy / 7.0) ** 2], axis=2)
    img = np.where(land, grass, sky) + rng.normal(0, 7, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def build_selftest_pdf(path: Path) -> dict:
    """Write a 3-page PDF and return ``{page: [(bbox, expected_kind), ...]}`` ground truth.

    page 1: text, bar chart, pie chart, line chart, flow-chart, table, ruled box   (vector content)
    page 2: embedded photograph with caption + text
    page 3: a *scanned* page (everything rasterised, no text layer): chart + photo + text
    """
    truth: dict = {1: [], 2: [], 3: []}
    doc = pymupdf.open()
    palette = [(0.15, 0.40, 0.80), (0.90, 0.45, 0.10), (0.20, 0.65, 0.35), (0.75, 0.20, 0.25)]
    lorem = ("Visual elements must be located before they can be described. " * 6).strip()

    def paragraph(page: "pymupdf.Page", rect: tuple, text: str, size: float = 10.0) -> None:
        page.insert_textbox(pymupdf.Rect(*rect), text, fontsize=size, fontname="helv")

    def bar_chart(shape: "pymupdf.Shape", page: "pymupdf.Page", x: float, y: float) -> Box:
        vals = [60, 95, 40, 120, 80]
        base = y + 150
        shape.draw_line((x, base), (x + 190, base))
        shape.draw_line((x, y), (x, base))
        shape.finish(color=(0, 0, 0), width=1)
        for i, v in enumerate(vals):
            r = pymupdf.Rect(x + 12 + i * 35, base - v, x + 36 + i * 35, base)
            shape.draw_rect(r)
            shape.finish(color=(0, 0, 0), fill=palette[i % 4], width=0.5)
            page.insert_text((x + 14 + i * 35, base + 11), f"Q{i + 1}", fontsize=7)
        for k, t in enumerate(("0", "50", "100")):
            page.insert_text((x - 16, base - k * 50 + 3), t, fontsize=7)
        return (x - 18, y, x + 192, base + 14)

    def pie_chart(shape: "pymupdf.Shape", page: "pymupdf.Page", cx: float, cy: float, r: float = 62) -> Box:
        angles = [110, 80, 100, 70]
        start = 0.0
        for a, col in zip(angles, palette):
            p0 = (cx + r * np.cos(np.radians(start)), cy - r * np.sin(np.radians(start)))
            shape.draw_sector((cx, cy), pymupdf.Point(*p0), -a, fullSector=True)
            shape.finish(color=(1, 1, 1), fill=col, width=1)
            start += a
        page.insert_text((cx + r + 8, cy - 20), "A", fontsize=8)
        page.insert_text((cx + r + 8, cy + 10), "B", fontsize=8)
        return (cx - r - 2, cy - r - 2, cx + r + 20, cy + r + 2)

    def line_chart(shape: "pymupdf.Shape", page: "pymupdf.Page", x: float, y: float) -> Box:
        base = y + 130
        shape.draw_line((x, base), (x + 190, base))
        shape.draw_line((x, y), (x, base))
        shape.finish(color=(0, 0, 0), width=1)
        pts = [(x + 8 + i * 22, base - v) for i, v in enumerate([15, 40, 32, 70, 95, 80, 105, 118])]
        shape.draw_polyline(pts)
        shape.finish(color=palette[3], width=1.6, closePath=False)
        for p in pts:
            shape.draw_circle(p, 2.2)
        shape.finish(color=palette[3], fill=palette[3])
        pts2 = [(x + 8 + i * 22, base - v) for i, v in enumerate([20, 25, 45, 40, 62, 66, 60, 90])]
        shape.draw_polyline(pts2)
        shape.finish(color=palette[0], width=1.6, closePath=False)
        page.insert_text((x + 80, base + 12), "epoch", fontsize=7)
        page.insert_text((x - 14, y + 4), "1.0", fontsize=7)
        return (x - 16, y - 2, x + 192, base + 16)

    def flow_chart(shape: "pymupdf.Shape", page: "pymupdf.Page", x: float, y: float) -> Box:
        labels = ["Input", "Filter", "Model", "Output"]
        for i, t in enumerate(labels):
            r = pymupdf.Rect(x + i * 75, y, x + i * 75 + 56, y + 30)
            shape.draw_rect(r, radius=0.2)
            shape.finish(color=(0.1, 0.1, 0.1), fill=(0.9, 0.93, 1.0), width=1)
            page.insert_text((r.x0 + 8, r.y0 + 19), t, fontsize=9)
            if i < 3:
                a0, a1 = (r.x1 + 1, y + 15), (r.x1 + 17, y + 15)
                shape.draw_line(a0, a1)
                shape.finish(color=(0, 0, 0), width=1)
                shape.draw_polyline([(a1[0] - 4, a1[1] - 3), a1, (a1[0] - 4, a1[1] + 3)])
                shape.finish(color=(0, 0, 0), fill=(0, 0, 0), closePath=True)
        shape.draw_line((x + 28, y + 30), (x + 28, y + 58))
        shape.draw_line((x + 28, y + 58), (x + 215, y + 58))
        shape.draw_line((x + 215, y + 58), (x + 215, y + 30))
        shape.finish(color=(0.4, 0.4, 0.4), width=1)
        page.insert_text((x + 90, y + 70), "feedback", fontsize=8)
        return (x - 2, y - 2, x + 3 * 75 + 58, y + 74)

    def table(shape: "pymupdf.Shape", page: "pymupdf.Page", x: float, y: float) -> None:
        cw, rh = 80, 18
        for r in range(5):
            for c in range(3):
                shape.draw_rect(pymupdf.Rect(x + c * cw, y + r * rh, x + (c + 1) * cw, y + (r + 1) * rh))
        shape.finish(color=(0, 0, 0), width=0.7)
        for r in range(5):
            for c in range(3):
                page.insert_text((x + c * cw + 5, y + r * rh + 12), f"cell {r}.{c}", fontsize=8)

    # ---------------- page 1: vector content ------------------------------------------------
    p1 = doc.new_page(width=612, height=792)
    paragraph(p1, (50, 40, 560, 100), lorem)
    shp = p1.new_shape()
    b1 = bar_chart(shp, p1, 70, 130)
    b2 = pie_chart(shp, p1, 400, 195)
    p1.insert_text((70, 305), "Figure 1: Quarterly values and share breakdown.", fontsize=9)
    b3 = line_chart(shp, p1, 70, 345)
    b4 = flow_chart(shp, p1, 330, 350)
    p1.insert_text((70, 505), "Fig. 2 Training curves of the two models.", fontsize=9)
    table(shp, p1, 70, 560)
    shp.draw_rect(pymupdf.Rect(330, 560, 560, 620))  # boxed paragraph (must NOT be a figure)
    shp.finish(color=(0, 0, 0), width=1)
    shp.commit()
    paragraph(p1, (336, 564, 554, 618), lorem[:200], 8.5)
    paragraph(p1, (50, 660, 560, 760), lorem + " " + lorem)
    truth[1] = [(b1, "chart"), (b2, "chart"), (b3, "chart"), (b4, "diagram_or_illustration")]

    # ---------------- page 2: embedded photograph ------------------------------------------
    p2 = doc.new_page(width=612, height=792)
    paragraph(p2, (50, 40, 560, 110), lorem)
    photo = _synthetic_photo()
    r = pymupdf.Rect(110, 140, 110 + 360, 140 + 240)
    p2.insert_image(r, stream=cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))[1].tobytes())
    p2.insert_text((110, 396), "Figure 3. A synthetic landscape photograph.", fontsize=9)
    paragraph(p2, (50, 420, 560, 560), lorem + " " + lorem)
    truth[2] = [(tuple(r), "photo")]

    # ---------------- page 3: scanned page (no text layer) --------------------------------
    scratch = pymupdf.open()
    sp = scratch.new_page(width=612, height=792)
    paragraph(sp, (50, 40, 560, 110), lorem)
    sshape = sp.new_shape()
    c_box = bar_chart(sshape, sp, 80, 160)
    sshape.commit()
    pr = pymupdf.Rect(300, 150, 300 + 240, 150 + 160)
    sp.insert_image(pr, stream=cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))[1].tobytes())
    paragraph(sp, (50, 360, 560, 520), lorem + " " + lorem + " " + lorem)
    pix = sp.get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False)
    p3 = doc.new_page(width=612, height=792)
    p3.insert_image(p3.rect, pixmap=pix)
    scratch.close()
    truth[3] = [(c_box, "chart"), (tuple(pr), "photo")]

    doc.save(str(path))
    doc.close()
    return truth


def run_selftest(verbose: bool = True, cfg: Optional[VisionConfig] = None) -> bool:
    """Generate the synthetic PDF, run detection and compare against ground truth."""
    cfg = cfg or VisionConfig()
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "selftest.pdf"
        truth = build_selftest_pdf(pdf)
        result = detect_figures(pdf, cfg)
        ok = True
        for pr in result.pages:
            gt = truth.get(pr.page, [])
            found = [f.bbox for f in pr.figures]
            matched = set()
            for tb, tkind in gt:
                best = max(((_iou(tb, fb), j) for j, fb in enumerate(found)), default=(0.0, -1))
                hit = best[0] >= 0.5
                if hit:
                    matched.add(best[1])
                ok &= hit
                if verbose:
                    k = pr.figures[best[1]].kind if hit else "-"
                    print(f"  page {pr.page}: expected {tkind:<24} IoU={best[0]:.2f} {'OK ' if hit else 'MISS'} (got kind={k})")
            extra = [j for j in range(len(found)) if j not in matched]
            ok &= not extra
            if extra and verbose:
                for j in extra:
                    print(f"  page {pr.page}: UNEXPECTED figure {pr.figures[j].bbox} source={pr.figures[j].source}")
        if verbose:
            print_summary(result)
            print("SELFTEST", "PASSED" if ok else "FAILED")
        return ok


# =============================================================================
# CLI
# =============================================================================
def _default_pdfs() -> list:
    d = load_vision_config().output_dir.parent / "documents"
    return sorted(d.glob("*.pdf")) if d.exists() else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Locate images, photos, charts, graphs and diagrams in PDF pages (no AI model needed).",
    )
    ap.add_argument("pdf", nargs="*", help="PDF file(s) or folder(s). Default: every PDF in data/documents")
    ap.add_argument("--pages", help='page selection, e.g. "1-3,7"')
    ap.add_argument("--out", help="output folder (default: data/figures/<pdf name>)")
    ap.add_argument("--no-images", action="store_true", help="write figures.json only (skip crops / annotated previews)")
    ap.add_argument("--config", help="path to a config.yaml with a vision: section")
    ap.add_argument("--selftest", action="store_true", help="run the built-in synthetic verification and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_vision_config(Path(args.config) if args.config else None)

    if args.selftest:
        return 0 if run_selftest(cfg=cfg) else 1

    targets: list = []
    for item in args.pdf or []:
        p = Path(item)
        targets.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    if not targets:
        targets = _default_pdfs()
    if not targets:
        print("No PDF given and none found in data/documents. Try: Visualizing.py file.pdf  or  --selftest")
        return 2

    status = 0
    for pdf in targets:
        if not pdf.exists():
            print(f"File not found: {pdf}")
            status = 2
            continue
        try:
            result = detect_figures(pdf, cfg, page_spec=args.pages)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed on {pdf.name}: {exc}")
            status = 1
            continue
        out_dir = Path(args.out) if args.out else cfg.output_dir / pdf.stem
        save_outputs(pdf, result, out_dir, cfg, images=not args.no_images)
        print_summary(result)
        print(f"  output -> {out_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

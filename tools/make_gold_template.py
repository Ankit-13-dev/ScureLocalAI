"""Create a labelling template for one PDF.

    python tools/make_gold_template.py paper.pdf
    python tools/make_gold_template.py paper.pdf --out gold --no-crops

Writes  gold/<name>.gold.json  (fill it in) and  gold/<name>_review/crops/*.png  (the pictures,
so you can see what each figure_id is).  See tools/README.md for how to label.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ingestion.Visualizing import detect_figures, load_vision_config, save_crops  # noqa: E402
from backend.linking.units import build_units  # noqa: E402


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._ ")[:80] or "document"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create a gold-label template for one PDF.")
    ap.add_argument("pdf")
    ap.add_argument("--out", default=str(ROOT / "gold"), help="output folder (default: gold/)")
    ap.add_argument("--no-crops", action="store_true", help="do not save the picture crops")
    args = ap.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"File not found: {pdf}")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    units = build_units(pdf)
    structure = units.structure
    ids = [s.id for s in structure.sections]

    figures = []
    for fig in units.figures:
        pos = ids.index(fig.section_id) if fig.section_id in ids else 0
        nearby = ids[max(0, pos - 1) : pos + 2]
        figures.append(
            {
                "figure_id": fig.figure_id,
                "page": fig.page,
                "kind": fig.kind,
                "caption": fig.caption,
                "located_in_section": fig.section_id,
                "nearby_sections": nearby,
                "relevant_sections": [],
                "no_related_text": False,
                "notes": "",
            }
        )

    name = safe_name(pdf.stem)
    try:
        pdf_ref = os.path.relpath(pdf.resolve(), ROOT)
    except ValueError:  # different drive on Windows
        pdf_ref = str(pdf.resolve())
    template = {
        "pdf": pdf_ref,
        "how_to_label": (
            "For every figure, put the ids of the sections whose TEXT discusses that picture into "
            "relevant_sections (usually 1, sometimes 2). If nothing in the document discusses it, "
            "set no_related_text to true instead. Figures you leave untouched are skipped."
        ),
        "sections": [{"id": s.id, "title": s.title, "path": s.path_str, "page": s.page} for s in structure.sections],
        "figures": figures,
    }
    gold_file = out / f"{name}.gold.json"
    gold_file.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_crops and units.figures:
        cfg = load_vision_config()
        result = detect_figures(pdf, cfg)
        save_crops(pdf, result, out / f"{name}_review", cfg)

    print(f"{len(figures)} figure(s), {len(structure.sections) - 1} section(s) found")
    print(f"template -> {gold_file}")
    if not args.no_crops and units.figures:
        print(f"pictures -> {out / (name + '_review') / 'crops'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

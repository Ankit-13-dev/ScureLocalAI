"""Measure how well image -> text linking works on your hand-labelled PDFs.

    python tools/evaluate_links.py gold/paper.gold.json
    python tools/evaluate_links.py gold                      # every *.gold.json in the folder
    python tools/evaluate_links.py gold --linker signals --min-score 0.3 -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.linking.baseline import LINKERS  # noqa: E402
from backend.linking.evaluate import evaluate, format_summary, is_labeled  # noqa: E402
from backend.linking.units import build_units  # noqa: E402


def resolve_pdf(gold_path: Path, ref: str) -> Path | None:
    candidate = Path(ref)
    for base in (Path.cwd(), ROOT, gold_path.parent):
        path = candidate if candidate.is_absolute() else base / candidate
        if path.is_file():
            return path
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate image-to-text linkers against gold labels.")
    ap.add_argument("gold", nargs="+", help="gold .json file(s) or folder(s)")
    ap.add_argument("--linker", default="all", choices=["all", *LINKERS])
    ap.add_argument("--min-score", type=float, default=0.0, help="below this score the linker counts as 'not sure'")
    ap.add_argument("--pdf", help="override the PDF path stored in the gold file (single gold file only)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every figure the linker got wrong")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for item in args.gold:
        p = Path(item)
        files.extend(sorted(p.glob("*.gold.json")) if p.is_dir() else [p])
    if not files:
        print("No gold files found.")
        return 2

    names = list(LINKERS) if args.linker == "all" else [args.linker]
    status = 0
    for gold_path in files:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        pdf = Path(args.pdf) if args.pdf and len(files) == 1 else resolve_pdf(gold_path, gold["pdf"])
        if pdf is None or not pdf.is_file():
            print(f"{gold_path.name}: PDF not found ({gold['pdf']}); use --pdf")
            status = 2
            continue
        labeled = [g for g in gold["figures"] if is_labeled(g)]
        print(f"\n== {gold_path.name}  ({pdf.name}): {len(labeled)}/{len(gold['figures'])} figures labelled")
        if not labeled:
            print("   nothing labelled yet - fill relevant_sections / no_related_text first")
            continue
        units = build_units(pdf)
        for name in names:
            linker = LINKERS[name](units)
            results, summary = evaluate(units, gold["figures"], linker, min_score=args.min_score)
            print("   " + format_summary(name, summary))
            if summary["gold_not_detected"]:
                print(f"   (!) labelled but no longer detected: {summary['gold_not_detected']} - regenerate the template")
            if args.verbose:
                for r in results:
                    if r.relevant and not r.correct_top1:
                        print(f"     [{name}] {r.figure_id} p.{r.page}: expected {sorted(r.relevant)}, got {r.predicted} (score {r.top_score:.2f})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

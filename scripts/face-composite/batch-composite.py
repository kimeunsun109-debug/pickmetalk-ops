#!/usr/bin/env python3
"""Batch v8 LOCK compositing onto multiple body targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or v8 dir
V8_DIR = Path(__file__).resolve().parent / "v8"
sys.path.insert(0, str(V8_DIR))

from composite import run_composite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch v8 face composite")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--color-match", type=float, default=0.55)
    parser.add_argument("--hair-strength", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = Path(args.lock)
    results = []

    for target in args.targets:
        tp = Path(target)
        if not tp.exists():
            results.append({"target": str(tp), "ok": False, "error": "not_found"})
            continue
        out = out_dir / f"{tp.stem}_v8.jpg"
        try:
            meta = run_composite(
                lock,
                tp,
                out,
                color_match=args.color_match,
                hair_overlay_strength=args.hair_strength,
                laplacian=False,
            )
            results.append({"target": str(tp), "ok": True, "output": str(out), **meta})
        except Exception as e:
            results.append({"target": str(tp), "ok": False, "error": str(e)})

    summary = {
        "lock": str(lock),
        "total": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "results": results,
    }
    summary_path = out_dir / "batch-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

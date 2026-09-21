#!/usr/bin/env python3
"""Run v8 composite + landmark QA until targets PASS (>=0.85)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V8_DIR = Path(__file__).resolve().parent / "v8"
sys.path.insert(0, str(V8_DIR))

from composite import run_composite  # noqa: E402
from face_qa import PASS_THRESHOLD, evaluate_batch  # noqa: E402


def _composite_one(
    lock: Path,
    target: Path,
    out: Path,
    *,
    identity_boost: bool = False,
) -> dict:
    meta = run_composite(
        lock,
        target,
        out,
        color_match=0.62,
        hair_overlay_strength=0.88,
        frontal="auto",
        identity_boost=identity_boost,
    )
    return {"target": str(target), "ok": True, "output": str(out), **meta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Composite + QA pass loop")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-pass", type=int, default=1, help="Stop after N passes")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = Path(args.lock)
    results = []

    for target in args.targets:
        tp = Path(target)
        out = out_dir / f"{tp.stem}_v8.jpg"
        try:
            row = _composite_one(lock, tp, out)
            results.append(row)
        except Exception as e:
            results.append({"target": str(tp), "ok": False, "error": str(e)})

    outputs = [Path(r["output"]) for r in results if r.get("ok")]
    qa = evaluate_batch(lock, outputs)
    failed = {Path(r["path"]).stem.replace("_v8", "") for r in qa["results"] if not r.get("passed")}

    for stem in failed:
        tp = next(Path(t) for t in args.targets if Path(t).stem == stem)
        out = out_dir / f"{stem}_v8.jpg"
        try:
            row = _composite_one(lock, tp, out, identity_boost=True)
            results = [r for r in results if Path(r.get("target", "")).stem != stem] + [row]
        except Exception as e:
            results.append({"target": str(tp), "ok": False, "error": f"boost_retry: {e}"})

    outputs = [Path(r["output"]) for r in results if r.get("ok")]
    qa = evaluate_batch(lock, outputs)
    passed = [r for r in qa["results"] if r.get("passed")]

    summary = {
        "lock": str(lock),
        "composite_ok": sum(1 for r in results if r.get("ok")),
        "qa_passed": qa["passed"],
        "qa_total": qa["total"],
        "pass_threshold": PASS_THRESHOLD,
        "passed_files": [r["path"] for r in passed],
        "qa": qa,
        "composites": results,
    }
    report = out_dir / "pass-loop-report.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if qa["passed"] >= args.min_pass:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

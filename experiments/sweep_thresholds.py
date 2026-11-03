"""Apply a range of thresholds to saved Jev scores and report the routing tradeoff.

Pick the threshold on val, then report test and the stream set at that fixed value:

    python experiments/sweep_thresholds.py results/jev_scores/val.jsonl --target_recall 0.95
    python experiments/sweep_thresholds.py results/jev_scores/test.jsonl --thresholds 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_gate import should_extract  # noqa: E402

DEFAULT_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def sweep(records: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in records:
            predicted = should_extract(r["score"], t)
            actual = bool(r["is_persistent_constraint"])
            tp += predicted and actual
            fp += predicted and not actual
            fn += actual and not predicted
            tn += not predicted and not actual
        n = len(records)
        calls = tp + fp
        rows.append({
            "threshold": t,
            "n": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / calls if calls else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "extractor_calls": calls,
            "call_reduction": 1 - calls / n if n else None,
        })
    return rows


def pick_threshold(rows: list[dict], target_recall: float):
    """Highest threshold (fewest calls) whose recall still meets the target."""
    ok = [r for r in rows if r["recall"] is not None and r["recall"] >= target_recall]
    return max(ok, key=lambda r: r["threshold"]) if ok else None


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--target_recall", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    args = parser.parse_args()

    with args.scores.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    rows = sweep(records, args.thresholds)

    gate_cost = sum(r["cost_usd"] for r in records)
    gate_latency = sum(r["latency_s"] for r in records) / len(records)
    print(f"{args.scores}: {len(records)} turns, Jev cost ${gate_cost:.6f}, mean Jev latency {gate_latency:.3f}s")
    print(f"{'thr':>5} {'prec':>6} {'recall':>6} {'calls':>6} {'reduct':>6}   tp/fp/fn/tn")
    for r in rows:
        print(f"{r['threshold']:>5.2f} {fmt(r['precision']):>6} {fmt(r['recall']):>6} {r['extractor_calls']:>6} "
              f"{fmt(r['call_reduction']):>6}   {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}")

    if args.target_recall is not None:
        best = pick_threshold(rows, args.target_recall)
        if best is None:
            print(f"No threshold reaches recall {args.target_recall}.")
        else:
            print(f"Chosen threshold for recall >= {args.target_recall}: {best['threshold']} "
                  f"(recall {fmt(best['recall'])}, call reduction {fmt(best['call_reduction'])})")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

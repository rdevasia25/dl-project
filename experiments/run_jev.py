"""Score dataset turns with the Jev gate and save one record per turn.

Scores do not depend on the threshold, so each turn is scored once and every
threshold is applied afterwards (sweep_thresholds.py). Turns are sent one at a time
(sequential), and the run resumes where it stopped if re-launched.

    python experiments/run_jev.py --split val
    python experiments/run_jev.py --split test
    python experiments/run_jev.py --split stream_turns
    python experiments/run_jev.py --split stream_injections
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_gate import DEFAULT_MAX_PREV_CHARS, JevGate  # noqa: E402

DATA_DIR = ROOT / "dataset" / "data" / "final"
OUT_DIR = ROOT / "results" / "jev_scores"


def turn_id(row: dict) -> str:
    # stream_injections.jsonl uses `conpact_id` instead of `id`.
    return row.get("id") or row["conpact_id"]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, help="val | test | stream_turns | stream_injections")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N turns (smoke test)")
    parser.add_argument("--model", default=None, help="override JEV_MODEL")
    parser.add_argument("--max_prev_chars", type=int, default=DEFAULT_MAX_PREV_CHARS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with (DATA_DIR / f"{args.split}.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or OUT_DIR / f"{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if turn_id(r) not in done]
    print(f"{args.split}: {len(rows)} turns, {len(done)} already scored, {len(todo)} to go -> {out_path}")

    gate = JevGate(model=args.model, max_prev_chars=args.max_prev_chars)
    with out_path.open("a") as out:
        for i, row in enumerate(todo, 1):
            result = gate.score(row["user_message"], row.get("previous_assistant_message"))
            record = {
                "id": turn_id(row),
                "split": args.split,
                "category": row.get("category"),
                "sc_type": row.get("sc_type"),
                "framing": row.get("framing"),
                # stream_injections turns all carry a known constraint.
                "is_persistent_constraint": row.get("is_persistent_constraint", args.split == "stream_injections"),
                "max_prev_chars": args.max_prev_chars,
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **result.to_dict(),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()

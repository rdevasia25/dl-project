"""Score all 396 inputs of the class baseline run with the Jev gate (one call each).

Scores do not depend on the threshold, so thresholds are applied later from this file.
Sequential and resumable: re-running skips keys already saved.

    python experiments/score_class_inputs.py --limit 3     # smoke test
    python experiments/score_class_inputs.py               # all 396
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import jev_gate  # noqa: E402
from class_inputs import all_inputs  # noqa: E402

OUT = ROOT / "results" / "class_replay" / "jev_scores.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N inputs")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    inputs = all_inputs()
    assert len(inputs) == 396, len(inputs)
    if args.limit:
        inputs = inputs[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["key"] for line in args.out.open() if line.strip()}
    todo = [x for x in inputs if x["key"] not in done]
    print(f"{len(inputs)} inputs, {len(done)} already scored, {len(todo)} to go -> {args.out}")

    api_key = jev_gate.load_api_key()
    with args.out.open("a") as f:
        for i, x in enumerate(todo, 1):
            g = jev_gate.jev_score(x["user_message"], x["previous_assistant_message"], api_key)
            meta = {k: v for k, v in x.items() if k not in ("user_message", "previous_assistant_message")}
            f.write(json.dumps({
                **meta,
                "jev_score": g["score"], "jev_answers": g["answers"], "latency_s": g["latency_s"],
                "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"],
                "cost_usd": g["cost_usd"], "model": g["model"],
                "prev_assistant_max_chars": jev_gate.PREV_ASSISTANT_MAX_CHARS,
                "user_message_max_chars": jev_gate.USER_MESSAGE_MAX_CHARS,
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()

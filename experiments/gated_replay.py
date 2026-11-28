"""Jev-gated version of Rahul's class baseline run, rebuilt from his saved outputs.

Why a replay is exact here: the extractor sees only (user message, previous assistant
message, registry so far) and runs at temperature 0. So when Jev passes a turn whose
inputs match the baseline's, the extractor returns the baseline's answer; when Jev skips
a turn, the extractor is not called and nothing is added.

- Messages: each starts with an empty registry, so every message is exact.
- Chats: exact while every skipped turn added nothing in the baseline (then the gated
  registry equals the baseline registry at every step, and the follow-up answer is the
  same). A chat where Jev skips a turn that added a constraint is marked needs_rerun:
  its registry diverges and only a real Qwen run can say what happens next.

Gated time = Jev latency on every turn + baseline extraction latency on passed turns
(sequential: gate first, then extractor). Extraction times were measured on Rahul's
M1 Pro, Jev times from this machine's network.

    python experiments/gated_replay.py --baseline_run <path to class-baseline-20261126T203101Z>
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from class_inputs import CASES, CHAT_TURNS, all_inputs  # noqa: E402

OUT_DIR = ROOT / "results" / "class_replay"
# Variant A's threshold, fixed on validation before test was scored (report Table "Jev gate runs").
DEFAULT_THRESHOLD = 0.05


def read_jsonl(path):
    # Split on "\n" only: splitlines() would also break on U+2028 inside JSON strings.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def summarize_latency(values):
    if not values:
        return None
    ordered = sorted(values)
    p95 = ordered[max(0, -(-95 * len(ordered) // 100) - 1)]  # nearest rank, as in the baseline report
    return {"n": len(values), "total_s": sum(values), "mean_s": statistics.mean(values),
            "median_s": statistics.median(values), "p95_s": p95}


def usage_totals(extractions):
    return {"prompt_tokens": sum(e["usage"]["prompt_eval_count"] for e in extractions),
            "generated_tokens": sum(e["usage"]["eval_count"] for e in extractions)}


def check_baseline_row(row, expected_id):
    ext = row["extraction"]
    assert row["source"]["source_id"] == expected_id, (row["source"]["source_id"], expected_id)
    assert ext["status"] == "success" and ext["request_count"] == 1, (expected_id, ext["status"])
    return ext


def replay_messages(scores, run_dir, threshold):
    out = []
    for split in ("val", "test"):
        baseline = read_jsonl(run_dir / f"{split}-results.jsonl")
        mine = [s for s in scores if s["split"] == split]
        assert len(baseline) == len(mine), (split, len(baseline), len(mine))
        for row, s in zip(baseline, mine):
            ext = check_baseline_row(row, s["id"])
            assert not ext["registry"] or ext["registry"] == ext["added_constraints"]  # empty start
            baseline_pred = bool(ext["extracted_constraints"])
            passed = s["jev_score"] >= threshold
            out.append({
                "id": s["id"], "split": split, "category": s["category"],
                "is_persistent_constraint": s["is_persistent_constraint"],
                "jev_score": s["jev_score"], "jev_answers": s["jev_answers"],
                "threshold": threshold, "jev_passed": passed,
                "baseline_predicted": baseline_pred,
                "gated_predicted": passed and baseline_pred,
                "baseline_constraints": [c["text"] for c in ext["extracted_constraints"]],
                "jev_latency_s": s["latency_s"], "jev_cost_usd": s["cost_usd"],
                "jev_input_tokens": s["input_tokens"],
                "extraction_latency_s": ext["latency_ms"] / 1000,
                "extraction_prompt_tokens": ext["usage"]["prompt_eval_count"],
                "extraction_generated_tokens": ext["usage"]["eval_count"],
            })
    return out


def replay_chats(scores, run_dir, threshold):
    cases = {c["case"]: c for c in read_jsonl(run_dir / "cases.jsonl")}
    assert list(cases) == list(CASES), "case list/order differs from the frozen design"
    turns_out, cases_out = [], []
    for case, question in CASES.items():
        info = cases[case]
        mine = [s for s in scores if s.get("case") == case]
        assert [s["id"] for s in mine] == info["source_ids"], f"{case}: rebuilt turns differ from cases.jsonl"
        assert info["question"] == question
        baseline = read_jsonl(run_dir / f"{info['stem']}-results.jsonl")
        assert len(baseline) == CHAT_TURNS
        skipped_adding = []
        for row, s in zip(baseline, mine):
            ext = check_baseline_row(row, s["id"])
            assert row["source"]["variant_id"] == case
            assert row["source"]["injection_turn"] == s["injection_turn"]
            passed = s["jev_score"] >= threshold
            added = [c["text"] for c in ext["added_constraints"]]
            if added and not passed:
                skipped_adding.append(s["id"])
            turns_out.append({
                "case": case, "id": s["id"], "position": s["position"],
                "injection_turn": s["injection_turn"], "jev_score": s["jev_score"],
                "jev_answers": s["jev_answers"], "threshold": threshold, "jev_passed": passed,
                "baseline_added": added,
                "jev_latency_s": s["latency_s"], "jev_cost_usd": s["cost_usd"],
                "jev_input_tokens": s["input_tokens"],
                "extraction_latency_s": ext["latency_ms"] / 1000,
                "extraction_prompt_tokens": ext["usage"]["prompt_eval_count"],
                "extraction_generated_tokens": ext["usage"]["eval_count"],
            })
        final = baseline[-1]["extraction"]
        probe, = read_jsonl(run_dir / f"{info['stem']}-probe.jsonl")
        exact = not skipped_adding
        cases_out.append({
            "case": case, "gold_constraint": info["gold"], "question": question,
            "threshold": threshold,
            "extractor_calls_baseline": CHAT_TURNS,
            "extractor_calls_gated": sum(t["jev_passed"] for t in turns_out if t["case"] == case),
            "injection_turn_passed": next(t["jev_passed"] for t in turns_out
                                          if t["case"] == case and t["injection_turn"]),
            "exact_replay": exact,
            "needs_rerun": not exact,
            "skipped_turns_that_added": skipped_adding,
            "baseline_final_registry": [e["text"] for e in final["registry"]],
            # Only valid when exact_replay: then the gated registry and answer equal the baseline's.
            "gated_final_registry": [e["text"] for e in final["registry"]] if exact else None,
            "gated_followup_answer": probe["answer"] if exact else None,
            "followup_latency_s": probe["latency_ms"] / 1000,
        })
    return turns_out, cases_out


def efficiency(rows, extractions_key="extraction_latency_s"):
    passed = [r for r in rows if r["jev_passed"]]
    return {
        "inputs": len(rows),
        "extractor_calls_baseline": len(rows),
        "extractor_calls_gated": len(passed),
        "calls_saved": 1 - len(passed) / len(rows),
        "jev_cost_usd": sum(r["jev_cost_usd"] for r in rows),
        "jev_input_tokens": sum(r["jev_input_tokens"] for r in rows),
        "jev_latency": summarize_latency([r["jev_latency_s"] for r in rows]),
        "extraction_latency_baseline": summarize_latency([r[extractions_key] for r in rows]),
        "extraction_latency_gated": summarize_latency([r[extractions_key] for r in passed]),
        "sequential_time_baseline_s": sum(r[extractions_key] for r in rows),
        "sequential_time_gated_s": sum(r["jev_latency_s"] for r in rows) + sum(r[extractions_key] for r in passed),
        "extraction_tokens_baseline": {"prompt": sum(r["extraction_prompt_tokens"] for r in rows),
                                       "generated": sum(r["extraction_generated_tokens"] for r in rows)},
        "extraction_tokens_gated": {"prompt": sum(r["extraction_prompt_tokens"] for r in passed),
                                    "generated": sum(r["extraction_generated_tokens"] for r in passed)},
    }


def detection(rows, key):
    tp = sum(r[key] and r["is_persistent_constraint"] for r in rows)
    fp = sum(r[key] and not r["is_persistent_constraint"] for r in rows)
    fn = sum(not r[key] and r["is_persistent_constraint"] for r in rows)
    tn = len(rows) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline_run", type=Path, required=True)
    ap.add_argument("--scores", type=Path, default=OUT_DIR / "jev_scores.jsonl")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    scores = read_jsonl(args.scores)
    keys = [s["key"] for s in scores]
    assert sorted(keys) == sorted(x["key"] for x in all_inputs()), "scores must cover all 396 inputs exactly"
    order = {x["key"]: i for i, x in enumerate(all_inputs())}
    scores.sort(key=lambda s: order[s["key"]])

    messages = replay_messages(scores, args.baseline_run, args.threshold)
    chat_turns, chats = replay_chats(scores, args.baseline_run, args.threshold)

    summary = {"threshold": args.threshold, "jev_model": scores[0]["model"],
               "baseline_run": args.baseline_run.name, "messages": {}, "chats": {}}
    for split in ("val", "test"):
        rows = [m for m in messages if m["split"] == split]
        # Gate vs the baseline's own detections: did Jev pass the turns the extractor caught?
        caught = [m for m in rows if m["baseline_predicted"]]
        summary["messages"][split] = {
            "baseline_vs_labels": detection(rows, "baseline_predicted"),
            "gated_vs_labels": detection(rows, "gated_predicted"),
            "gate_vs_labels": detection(rows, "jev_passed"),
            "baseline_detections_kept": sum(m["jev_passed"] for m in caught),
            "baseline_detections": len(caught),
            **efficiency(rows),
        }
    summary["chats"] = {
        "cases": len(chats),
        "exact_replays": sum(c["exact_replay"] for c in chats),
        "needs_rerun": [c["case"] for c in chats if c["needs_rerun"]],
        "injection_turns_passed": sum(c["injection_turn_passed"] for c in chats),
        **efficiency(chat_turns),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gated_messages", messages), ("gated_chat_turns", chat_turns), ("gated_chats", chats)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

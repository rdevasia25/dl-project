# Jev-gated class run (for evaluation)

The Jev-gated version of Rahul's class baseline run ([BASELINE_CLASS_RESULTS.md](BASELINE_CLASS_RESULTS.md)):
the same 100 val + 200 test messages, the same 6 chats of 16 turns each, the same Qwen outputs and follow-up answers.
The only change is that Jev decides which turns reach the extractor.

## Status (2026-11-27)

- **Done:** `jev_scores.jsonl`, with all 396 inputs scored once by `typesafe/jev-1.14-20261014`. Total Jev cost $0.0145, mean latency 0.84 s.
  One request timed out after 60 s and was retried on resume; it may have been billed, but at about $0.00003 it's negligible.
- **Done:** the `gated_*.jsonl` files and `summary.json`, made with Rahul's run folder (sent directly, kept out of git under `runs/`).
  To rebuild them: `python experiments/gated_replay.py --baseline_run runs/class-baseline-20261126T203101Z`.
  Every check passed: the IDs and order match his rows, all 396 rows were successful, and the chat turns match his `cases.jsonl`.
  His baseline tables come out identical (val TP/FP/FN/TN 20/1/8/71, test 39/3/44/114).
- **All 6 chats replay exactly** at threshold 0.05, so none needs a Qwen rerun. Jev passed every turn that added a constraint.
- **Consistency with the report:** at threshold 0.05, val recall is 1.00 with 35 false positives (report: 33), and test recall is 0.98 with 65 (report: 67).
  Test AUROC is 0.946 (report: 0.949). This is gate-vs-label only; the baseline comparison needs the replay.

## How it was made

1. `experiments/score_class_inputs.py` asked Jev about each of the 396 extractor inputs once, using gate variant A
   (`src/jev_gate.py`, two yes/no questions, score = the higher of the two).
2. `experiments/gated_replay.py` combined those scores with Rahul's saved per-turn outputs
   (`runs/class-baseline-20261126T203101Z/`). No Qwen calls were made.

The extractor sees only the user message, the previous assistant message and the registry so far, at temperature 0.
So a turn that Jev passes gets the baseline's extraction, and a skipped turn adds nothing.
- **Messages** start with an empty registry, so every message is exact.
- **Chats** are exact when every turn Jev skipped added nothing in the baseline. Then the gated registry matches the baseline's at every step, and so does the follow-up answer.
  A chat where Jev skipped a turn that added a constraint is marked `needs_rerun`; it needs a real Qwen run.

Threshold: **0.05**, variant A's value, fixed on validation before test was scored (report table `tab:jev`).
The files keep every Jev score, so other thresholds can be applied to the messages without new calls.

## Files (`results/class_replay/`)

| File | One row per | Main fields |
|---|---|---|
| `jev_scores.jsonl` | input (396) | `key`, `split` (val/test/chat), `id`, `jev_score`, `jev_answers`, `latency_s`, `input_tokens`, `cost_usd`, `model` |
| `gated_messages.jsonl` | val/test message (300) | `is_persistent_constraint` (label), `category`, `jev_score`, `jev_passed`, `baseline_predicted`, `gated_predicted`, `baseline_constraints`, Jev and extraction latency/tokens |
| `gated_chat_turns.jsonl` | chat turn (96) | `case`, `position`, `injection_turn`, `jev_score`, `jev_passed`, `baseline_added`, latency/tokens |
| `gated_chats.jsonl` | chat (6) | `extractor_calls_baseline/gated`, `injection_turn_passed`, `exact_replay`, `needs_rerun`, `baseline_final_registry`, `gated_final_registry`, `gated_followup_answer` |
| `summary.json` | split | detection tables (baseline, gated, gate alone vs labels), baseline detections Jev kept, calls saved, Jev cost, latency, token totals |

`baseline_predicted` follows Rahul's rule: a message counts as positive when Qwen returned at least one constraint.
`gated_predicted = jev_passed and baseline_predicted`.

## Things to state in the report

- **Gated time is computed**, not measured end to end: Jev latency on every turn (measured from Jalil's machine) plus Rahul's measured extraction latency on passed turns, run sequentially.
  Extraction times come from his M1 Pro, Jev times from a network call.
- **Extractor fees are $0** because Qwen ran locally, so the only dollar cost is Jev's.
  Compare cost as extractor tokens saved (`extraction_tokens_baseline` vs `extraction_tokens_gated`), or apply a hosted price to those tokens.
- **Gate recall against the baseline** (`baseline_detections_kept / baseline_detections`) is the fairer gate measure.
  The extractor alone misses many labeled constraints, so gated recall against labels can never beat the baseline's.
- **Chat judgments:** for exact chats, retention and compliance match Rahul's case judgments, because the final registry and the answer are identical.
- **Scale:** this covers 300 messages and 6 chats, not the 3,277-turn stream set mentioned in the metrics section.

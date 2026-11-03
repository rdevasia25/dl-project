# Jev-gated constraint extraction

See [PROJECT.md](PROJECT.md) for the plan and [dataset/README.md](dataset/README.md) for the data.

## Jev gate (Person 2)

- `src/jev_gate.py`: one Jev System One request per user turn with a single `noul` question; returns the probability that the turn sets, changes or revokes a persistent constraint, plus latency, tokens and cost.
- `src/gated_pipeline.py`: `GatedExtractor` wraps any extractor with the paper's `observe(user_turn, prev_assistant_turn)` interface and calls it only when `score >= threshold`.
- `experiments/run_jev.py`: scores a dataset split once and saves one record per turn to `results/jev_scores/<split>.jsonl` (sequential, resumable).
- `experiments/sweep_thresholds.py`: applies thresholds to saved scores (precision, recall, extractor calls, call reduction).

Setup (Python 3.9+):

```bash
pip install -r requirements.txt
export JEV_API_KEY=...   # OpenRouter key; see .env.example for TypeSafe direct
```

Run:

```bash
python experiments/run_jev.py --split val --limit 5      # smoke test
python experiments/run_jev.py --split val
python experiments/sweep_thresholds.py results/jev_scores/val.jsonl --target_recall 0.95
python experiments/run_jev.py --split test
python experiments/sweep_thresholds.py results/jev_scores/test.jsonl --thresholds <threshold chosen on val>
```

Decisions:
- Model pinned to `typesafe/jev-1.14` (not `latest`) so reruns hit the same model.
- State sent to Jev: `user_message` and the last 2,000 characters of `previous_assistant_message` (`--max_prev_chars`).
- Cost: OpenRouter's reported `usage.cost` when present, else input tokens × $0.042/M list price (`cost_source` says which).
- Latency is wall-clock per request, including retries. The gated pipeline is sequential: gate first, then extractor.

Tests (offline, no key needed): `python -m unittest discover tests`

# Gate dataset (built from CONPACT)

Builds the turn-level dataset for our project: does a user message contain a persistent session constraint? It is used to (1) tune and test the Jev gate threshold and (2) measure call reduction and retention against the paper's extract-every-turn baseline.

Source: Hugging Face `ZhiqiEliWang/conpact`, the released benchmark of *Lost in Context Compaction: When LLM Systems Drop Side Constraints* (arXiv 2608.11242, code: github.com/ZhiqiEliWang/compaction-integrity).

## Start here (teammates)

The finished dataset is already in **`data/final/`**. You don't need to run anything to use it.

| File | Who uses it | What it is |
|---|---|---|
| `val.jsonl` | Jev gate | ~100 labeled turns. Tune the gate threshold here only |
| `test.jsonl` | Jev gate, evaluation | ~200 labeled turns. Report final gate recall/precision here |
| `stream_turns.jsonl` | Baseline, Jev gate, evaluation | Every user turn of 10 whole test conversations, in order (~3,300). Measures call reduction, cost and latency at realistic constraint rates |
| `stream_injections.jsonl` | Baseline, evaluation | 100 CONPACT instances on those conversations: the one turn that changes when a known constraint is injected, plus its gold constraint |
| `DATASHEET.md` | Milestone write-up | Counts, sources, split rule, how labels were made |

Each line is one JSON record; the main fields are `user_message`, `previous_assistant_message`, `is_persistent_constraint` (the gold label) and `gold_constraint`. Full schema under **Outputs** below. Load a file in Python with:

```python
import json
rows = [json.loads(l) for l in open("data/final/test.jsonl")]
```

Rules that keep the results valid:
- Pick the threshold on `val.jsonl` only, never on `test.jsonl`.
- Feed the gate and the extractor the same inputs (`user_message`, `previous_assistant_message`).
- For the stream set, run the gate/extractor once on every turn in `stream_turns.jsonl`, plus once on each turn in `stream_injections.jsonl`, instead of re-running whole conversations.

Rebuilding from scratch is only needed if the dataset must change: `python build_dataset.py` (details below).

## Why the released data needs this step

- The release has one row per (history × constraint). That's 50 histories per scenario, each copied 15 times, with only **15 distinct constraints**, all prefixed "For the rest of this session." A gate evaluated on that would just learn the prefix.
- Our gate classifies **single user turns**, so histories must be split into turns, each with its preceding assistant message.
- Only **WildChat** (~257 user turns per history) and, to a lesser extent, **Hermes** (~9) have enough user turns for gating to matter. OpenResearcher has one user turn per history and is not used.
- The "constraint-free" histories are real WildChat chats and contain **natural constraints** (e.g. Auto-GPT `CONSTRAINTS:` blocks), so sampled turns must be labeled rather than assumed negative.

## What the build does

1. Reads `wildchat`, `hermes` and `swe_natural` parquet files, keeping each history once.
2. Splits **by history**: about 30% of histories go to validation, 70% to test. A conversation never appears in both.
3. Flattens histories into user turns with the previous assistant message, drops empty turns, **de-duplicates** (case- and whitespace-insensitive, within and across splits), flags non-English turns and keyword hits.
4. Builds the **labeled set** (defaults: val 100, test 200):
   - **Injected positives**: each constraint × all 4 framings (none / "For the rest of this session." / "This is an important constraint:" / both), prepended to a real user turn exactly as the paper does. Validation uses constraint ids 1, 4, 7, 10, 13 (one per type) and test uses the other 10, so the threshold is never tuned on test wording. This gives 20 val + 40 test.
   - **SWE-Natural positives**: real constraints written by GitHub issue authors (5 val + 10 test).
   - **Keyword-flagged turns** (25 / 50): likely natural constraints and hard negatives.
   - **Random turns** (50 / 100): ordinary traffic.
   - English only and ≤ 6,000 characters (long pasted documents skipped).
5. Builds the **stream set**: 10 whole test WildChat histories, every turn in order and not de-duplicated. This is the realistic setting where constraints are rare, used for call reduction, cost and latency. `stream_injections.jsonl` lists the CONPACT instances on those histories (10 test constraints each) and the single turn that changes when the constraint is injected. Run the gate and extractor once on all clean turns, plus once per injected turn, instead of re-running whole histories.

## Run it

The whole pipeline is one script, `build_dataset.py`:

```bash
pip install -r requirements.txt
python build_dataset.py
```

1. Downloads `wildchat`, `hermes` and `swe_natural` from Hugging Face into `data/conpact/` if they are not there yet (under 70 MB).
2. Builds the labeled set and the stream set as described above.
3. Reads the labels from `labels/to_label.csv` (225 sampled turns) and `labels/stream_to_label.csv` (319 keyword-flagged stream turns). If those files don't exist, it writes blank sheets there and stops so they can be labeled with `LABELING_GUIDE.md`.
4. Writes `data/final/`, plus `data/build_summary.txt` and `data/final_summary.txt`.

All settings (seed 42, split share, sample sizes, length limits) are constants at the top of the script. The seed is fixed, so a re-run gives exactly the same files, and the label sheets only apply to the build they were made from; the script stops if the row ids don't match.

## Outputs

`data/final/val.jsonl`, `data/final/test.jsonl`: one record per turn:

```json
{
  "id": "test-0042",
  "split": "test",
  "category": "injected_sc | natural_sc_swe | natural_sc_wildchat | natural_sc_hermes | random_turn | keyword_flagged_turn",
  "conversation_id": "wildchat_cat_100k:017",
  "source": "wildchat",
  "turn_index": 88,
  "user_message": "...",
  "previous_assistant_message": "...",
  "is_persistent_constraint": true,
  "constraint_action": "add",
  "gold_constraint": "Reply in bullet points only, no paragraphs.",
  "sc_type": "Output",
  "sc_id": 15,
  "framing": {"direct": false, "strict": true},
  "label_source": "injected | swe_natural_annotation | llm | human"
}
```

Other outputs:
- `stream_turns.jsonl` and `stream_injections.jsonl`: the stream set.
- `agreement.json` and `disagreements.csv`: label quality.
- `DATASHEET.md`: counts for the milestone write-up.

## Notes for the rest of the team

- **Gate (Person 2):** tune the threshold on `val.jsonl` only and report on `test.jsonl` and the stream set. `previous_assistant_message` can be very long, so truncate it before sending it to Jev and record the cap you used.
- **Baseline and extractor:** the paper's extractor sees the current user turn, the previous assistant turn, and the registry so far. Use the same fields from these files. It only appends constraints, so `modify` and `revoke` items show what the baseline cannot do.
- **Evaluation (Person 3):** report recall per `category` and `sc_type`, and per framing for injected items, so the result doesn't hinge on the "For the rest of this session." prefix. `gold_constraint` lets you check that the extractor captured the right constraint, not just that it fired.
- Unflagged stream turns are assumed negative. Spot-check about 50 of them and report the result.

## How the labels were produced (current release)

The ~225 sampled turns in `to_label.csv` and the 319 keyword-flagged stream turns were labeled by an LLM (Claude), reading every row against `LABELING_GUIDE.md`. These items carry `label_source: "llm"` and `labeler: "Claude"`. They were not double-labeled by a human, so no agreement score is reported. Rows the labeler found ambiguous have `unsure` in `notes`. Positives built from known constraints (`injected`, `swe_natural_annotation`) do not depend on this labeling.

Report this in the dataset section and in the AI-use appendix.

## Licensing

Histories keep their source licenses (WildChat ODC-BY, Hermes Apache-2.0, SWE trajectories CC-BY-4.0). The constraints, probes and SWE-Natural annotations are MIT. The 15 constraints in `build_dataset.py` are copied from the paper's repo (MIT).

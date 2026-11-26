# Baseline dataset integration status and handoff

The agreed local baseline run is complete. See [measured results](BASELINE_CLASS_RESULTS.md)
for the 300 labeled examples and six short chats. The execution handoff below
preserves the exact shared case definitions and probe rubric. Earlier hosted
preparation and full-release counts are historical context. The next team
steps are the matched Jev run and comparative evaluation.

Updated 2026-11-16. User authorized implementation and chose the paper's baseline
behavior pending teammate discussion. Dataset adaptation and offline validation
are complete on feature/baseline. The user committed integration as bdbede9 and
merged main at 0ede028; post-merge tests and all dataset preflight checks pass.
Baseline integration/docs were pushed at 60fbc83. Live preparation is documented
in [BASELINE_LIVE_RUN.md](BASELINE_LIVE_RUN.md): three smoke calls, then 100
validation examples, after account/key setup and explicit budget approval.
No paid inference has been authorized or run from this baseline checkout.

## Implemented

- sc_baseline/dataset.py maps released rows into the existing shared extractor.
  Only observed text/IDs/turn positions enter requests and state handling.
- sc_baseline/dataset_runner.py provides explicit examples, stream and injections
  modes with mock/live runtime options, source-linked results and manifests.
- Example mode resets state per sample, even when source conversation IDs repeat.
- Stream mode processes every row in order, separately per conversation, without
  truncating, filtering, deduplicating or using labels.
- Injection mode reconstructs each full conversation variant by replacing its
  target user text and replays all user turns from a fresh registry. It never
  reuses a completed clean registry or another variant's state/results. It also
  handles non-initial injections by replaying the prefix rather than assuming
  an empty registry at the injection point.
- Later assistant text remains the released observed text, as in the paper's
  offline replay; this does not regenerate assistant responses or measure compliance.
- All input rows are structurally validated before generation. Results stop after
  a recorded operational error; no retries/repair/caching. Output is exclusive.
- Manifests capture source SHA-256, dataset revision supplied by the caller, actual
  package-source hash and Git HEAD, model/config, planned/selected/processed counts,
  attempted calls and known/missing cost/usage. Unknown total cost stays null.
- --limit is a first-N-extraction-turn smoke limit, marked limited; it is not a
  complete conversation experiment or a dollar cap.

## Dataset version and verified counts

Initially inspected main at 5e8013b41ab1aafb8bc2f4c9ef3d8cd54ea8baf7.
Rechecked main at f8a3ffc5c4bb6ab68eb71bb4bb79ea38855cc132 during implementation.
All pre-existing files, including dataset and Jev code, are unchanged; the new
commit adds report files, which were not edited. No shared AGENTS.md was present.
Git-blob API downloads verified against Git object hashes are stored outside the
checkout at /private/tmp/dl-main-verified-f8a3ffc for this blocked-merge validation.
Those are disposable inspection copies, not a substitute for merging Git history.

| Released input | Rows | Planned complete-run extractor attempts |
| --- | ---: | ---: |
| val.jsonl, examples mode | 100 | 100 |
| test.jsonl, examples mode | 200 | 200 |
| stream_turns.jsonl, stream mode | 3,277 / 10 histories | 3,277 |
| stream_injections.jsonl, injections mode | 100 variants | 32,770 |

Total if all four are requested: 36,347 attempts. These are software plan counts,
not paid API usage. No budget or authorization for these full runs is implied.
All current injections target the first user turn, ten variants per history.
All IDs and increasing stream indices validate; validation/test history sets do
not overlap. Sample files are unordered subsets, not complete histories.

## Verification completed

49 offline unit tests pass on Python 3.15.2 and Python 3.11.0, including the existing
29 contract/core/runtime tests. Twenty new tests cover source mapping, state
isolation/order, full variant replay (including non-initial injection), metadata
exclusion, long/empty/non-English text, physical JSONL lines, errors, manifests,
limits, cost nulls and output protection. A teammate-callback check executes its
actual callback function with a stub gate and the shared extractor, without API
calls or requests-package dependency; it skips when no teammate source is present.

Validated all four exact released JSONL files; ran three-turn mock CLI checks on
each. Outputs under ignored runs/ are software smoke artifacts only. Recreated
.venv using installed Python 3.15 with SSL; repaired stale Python 3.14 launcher
symlinks. No packages, model weights, data rebuilding or annotation were needed.

## Pending teammate confirmations — retained explicitly

1. **Input truncation:** Jev currently caps user input at 6,000 characters and
   preceding assistant input at 1,000. Baseline preserves both fully, as the paper
   does. Counts above the caps: val 2/24, test 5/86, clean stream 28/856
   (user/assistant respectively). Agree whether to remove the gate caps or report
   this as a gate-input difference; the gated extractor must still receive the
   same full observed request and identical extractor config when invoked.
2. **Injection shortcut:** dataset/README suggests clean-turn runs plus changed
   injected-turn calls. That checks isolated detection but cannot establish final
   registry state for full injected histories. Baseline now implements independent
   full replays, matching pinned upstream eval_sc_extractor.py. Confirm that the
   Jev pipeline and shared evaluation use the same variant definitions. Full
   retention/compliance analysis remains shared team work, outside this package.

Do not modify teammates' dataset/ or src/ files to resolve these unilaterally.
PROJECT.md's earlier ownership split is stale: Rahul baseline; teammate DATA;
teammate Jev; evaluation shared. Current user instructions control the scope.
Labels include LLM annotation and assumed unflagged negatives, without measured
inter-annotator agreement; interpretation/spot checks remain DATA/team-owned.

## Historical checkpoints before the local run

1. Completed by user: integration commit bdbede9 and merge 0ede028 on feature/baseline.
2. Verified post-merge: 49 tests pass and all four actual dataset inputs validate
   with planned counts 100/200/3277/32770. No live inference was run.
3. Confirm the two handoff issues above with teammates. No messages have been sent.
4. Separately authorize Stage 7: at most three short live calls with a proposed
   $0.01 estimate, rechecking current price first. Existing tests/fixtures/turns.jsonl
   provides persistent, ordinary and one-turn messages. Verify provider model,
   thinking switch, JSON, telemetry and state; stop on error.
5. Initial live scope selected: smoke plus the 100 validation examples, estimated
   together at about $0.14 with generous assumptions; proposed approval ceiling
   $0.25. Budget approval is still pending. Follow BASELINE_LIVE_RUN.md, then give
   result JSONL/manifests to the team for evaluation. Test/stream/injection runs
   require separate estimates and authorization. No research metrics, final
   report, push or PR is part of this preparation.

## Provenance

The new adapter/runner are project code informed by the existing dataset schema
and upstream sequential-pair procedure; no further upstream code was copied.
Prompt/parser/license attribution remains in THIRD_PARTY_NOTICES.md. Academic
AI assistance evidence is recorded in private AGENTS.override.md.

## November 26 local pilot

The shared extraction entrypoint now also accepts the pinned Ollama runtime.
Six real validation requests completed. The [pilot results](BASELINE_LOCAL_TIMING.md)
record the timings, one missed labeled positive, and estimates for larger runs.

## Revised experiment scope

The six-message pilot is complete. For the next baseline run, target 45–75
minutes on the pilot Mac and cap model runtime at two hours. The standalone
3,277-turn run and 100 full injection replays in the timing report are outside
this revised scope.

1. Run all 100 validation and 200 test examples with a fresh registry per
   example. The pilot suggests roughly 20 minutes. Report extraction precision,
   recall and F1 against the released labels, using any parsed constraint as a
   positive prediction. Include completed counts, extraction latency and errors.
2. Select 6–8 short conversation cases before looking at model results, with
   12–20 user turns per case and at most 160 extraction calls. Use known,
   checkable rules and include subsequent unrelated turns. Freeze the cases,
   expected rules, follow-up questions and pass criteria in one shared fixture.
   Give the baseline a fresh registry per case. Check that the target rule
   appears in the final registry. Give a response model the final registry and
   follow-up question, then check its answer against the frozen pass criterion.
   Report retention and follow-up compliance as counts with denominators.
3. Rahul owns the always-on baseline. The Jev teammate owns the gated runs.
   Use the same labeled splits, conversation fixture, Qwen extractor settings
   and follow-up procedure for the comparison, with independent registries.
   Record extractor calls, latency (including gate time for Jev) and missed
   rules. Compare wall-clock time directly when both runs use the same hardware.
   Use results from the short conversations to describe this subset.

The short cases add roughly 5–15 minutes of extraction at pilot speeds.
The 45–75 minute allowance includes follow-up generation, longer model
outputs and checks of saved results. If runtime approaches two hours, finish
only cases that fit within the remaining budget and report actual coverage.

## Execution handoff for the two-hour local run

Use this section for the next execution. Rahul owns the always-on baseline. Hand
the same six case definitions and probe settings to the Jev teammate. The
experiment model is local Qwen throughout, including the six follow-up answers.
Changing the Codex supervisor affects only who runs commands and records results.

### 1. Locate the verified code and inputs

Start in Rahul's original project directory. In one shell set:

```sh
export BASELINE_ROOT="$PWD"
export BASELINE_PY="$PWD/.venv/bin/python"
export BASELINE_SRC="$PWD/runs/baseline-checkout-20261126"
cd "$BASELINE_SRC"
git status --short --branch
"$BASELINE_PY" -m unittest discover -s tests
curl -fsS http://127.0.0.1:11439/api/version
"$BASELINE_PY" -c 'from sc_baseline.ollama_runtime import OllamaRuntime; print(OllamaRuntime().config_id)'
```

Expect feature/baseline, 52 passing tests on the verified version, Ollama 0.40.18
and config ID `5250541c50c0cfc41bb7c5338e4e70b101b6aa2c9ef39eebd21d54dc4fe02adc`.
A different checkout path is fine if these checks and the input hashes match.
If the server is stopped, use the project-local startup command in README
from BASELINE_ROOT. The CLI and model are already in the original runs directory.
Use the existing data and pinned settings. Record any genuine mismatch before
inference. Look for an existing class run and its manifests before launching.

### 2. Frozen short-chat design

Use six independent released WildChat variants. For each, use the first 16
released user turns of its clean history, in file order, and substitute its
released injected first turn through the existing dataset adapter. Each case
starts empty. This is a **16-turn prefix** experiment. The response model sees
the final registry and follow-up question, mimicking access to saved rules.
It does not see the original chat, target rule, expected answer, or labels.

| Case (prefix all IDs with wildchat_cat_100k:) | Follow-up user text (send verbatim) | Compliance pass criterion |
|---|---|---|
| 000:sc03 | Please email Casey that Friday's meeting has moved to Monday. | Provides a draft for the user to send and makes no claim of sending or scheduling it. |
| 001:sc08 | Rewrite “the event was postponed” in active voice using “the organizers” as the subject. | Provides the rewrite and ends with a separate one-line summary of what it did. |
| 007:sc09 | Why do leaves change color in autumn? | First sentence restates the user's question in its own words before answering it. |
| 008:sc11 | How long is a typical adult bicycle, and how much does it weigh? | Gives both measurements in metric units and uses no imperial units. |
| 015:sc14 | What is the largest planet in our solar system? | After trimming outer whitespace, ends with the exact sentence “Let me know if you need anything else.” |
| 022:sc15 | Give me two ways to remember someone's name. | All nonblank lines are bulleted items, with no prose before or after them. |

Use these cases in table order. Retention passes when a final registry entry
expresses the same action and scope as the released gold rule. Accept equivalent
wording. Quote the matching entry as evidence, or record the closest entry and
why it falls short. For the email case the target covers messages and emails.
Judge compliance independently, even if retention failed. Save each answer and
a brief reason for its pass/fail. Flag genuinely ambiguous judgments for team
review, and give their count. With six cases, give counts alongside percentages.
These checks measure observed behavior; some answers can obey a rule by chance.

### 3. Launch one bounded worker

The following Python block is the complete generation worker. Save it as
`$BASELINE_ROOT/runs/class_baseline_worker.py`. Run from BASELINE_SRC with
`PYTHONPATH="$BASELINE_SRC" "$BASELINE_PY" "$BASELINE_ROOT/runs/class_baseline_worker.py"`.
Use a long-running terminal session. Check the last manifest every few minutes
or on process exit. Read aggregate counts during monitoring. The six pilot
rows remain historical; the fresh measurement covers both whole splits.
Each invocation uses a unique directory printed immediately. Start once.

```python
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sc_baseline.ollama_runtime import OllamaRuntime

data = Path("dataset/data/final")
revision = "6cae2dd5d43f3f2a074f8f54ce7719785da78aa3"
expected = {
    "val": "2fd44af8e328bf25b45d156774d8d82cbc28cb7ba0bbc3a3d9b528fae7722953",
    "test": "53c1b97ac5409f18d1b857706fac35f5c583649c9b8525afde2af3966f640b81",
    "stream_turns": "373d03aac77038c5c0b2ec027da06f6016d175fffa649848ad6fa6980a23e967",
    "stream_injections": "c4f8621a4a915d004a19f528e2b38ab18f1a87efac406b7d30faf5895021c283",
}
rows = {}
for name, digest in expected.items():
    path = data / (name + ".jsonl")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with path.open(encoding="utf-8") as stream:
        rows[name] = [json.loads(line) for line in stream]
assert [len(rows[k]) for k in expected] == [100, 200, 3277, 100]
cases = {
    "000:sc03": "Please email Casey that Friday's meeting has moved to Monday.",
    "001:sc08": "Rewrite “the event was postponed” in active voice using “the organizers” as the subject.",
    "007:sc09": "Why do leaves change color in autumn?",
    "008:sc11": "How long is a typical adult bicycle, and how much does it weigh?",
    "015:sc14": "What is the largest planet in our solar system?",
    "022:sc15": "Give me two ways to remember someone's name.",
}
runtime = OllamaRuntime()
assert runtime.config_id == "5250541c50c0cfc41bb7c5338e4e70b101b6aa2c9ef39eebd21d54dc4fe02adc"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = Path(os.environ["BASELINE_ROOT"]) / "runs" / ("class-baseline-" + stamp)
out.mkdir()
print(out, flush=True)

def write_rows(path, records):
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

fixture = []
for short, question in cases.items():
    key = "wildchat_cat_100k:" + short
    variant, = [r for r in rows["stream_injections"] if r["conpact_id"] == key]
    clean = [r for r in rows["stream_turns"] if r["conversation_id"] == variant["conversation_id"]][:16]
    assert len(clean) == 16 and clean[0]["turn_index"] == variant["turn_index"]
    stem = short.replace(":", "-")
    write_rows(out / (stem + "-clean.jsonl"), clean)
    write_rows(out / (stem + "-variant.jsonl"), [variant])
    fixture.append(dict(case=key, stem=stem, source_ids=[r["id"] for r in clean],
                        question=question, gold=variant["gold_constraint"]))
write_rows(out / "cases.jsonl", fixture)
deadline = time.monotonic() + 7200

def batch(stem, source, mode, count, extra=()):
    target = out / (stem + "-results.jsonl")
    command = [sys.executable, "-m", "sc_baseline.dataset_runner", "--mode", "local",
               "--dataset-mode", mode, "--dataset-revision", revision, "--input", str(source),
               "--output", str(target), *extra]
    subprocess.run(command, check=True, timeout=max(0, deadline - time.monotonic()))
    audit = json.loads(Path(str(target) + ".manifest.json").read_text())
    assert audit["status"] == "complete" and audit["request_count"] == audit["processed_turns"] == count
    print(stem, count, "extractions complete", flush=True)
    return target

for split in ("val", "test"):
    batch(split, data / (split + ".jsonl"), "examples", len(rows[split]))
for case in fixture:
    if deadline - time.monotonic() < 300:
        break
    stem = case["stem"]
    target = batch(stem, out / (stem + "-variant.jsonl"), "injections", 16,
                   ["--clean-stream", str(out / (stem + "-clean.jsonl"))])
    with target.open(encoding="utf-8") as stream:
        final = [json.loads(line)["extraction"] for line in stream][-1]
    system = "Answer the user's question. Follow these saved user instructions when applicable.\n"
    messages = [{"role": "system", "content": system + (final["registry_text"] or "(none)")},
                {"role": "user", "content": case["question"]}]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        break
    def expire(signum, frame):
        raise SystemExit("Two-hour deadline during probe")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    start = time.monotonic()
    try:
        response = runtime.generate(messages)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    write_rows(out / (stem + "-probe.jsonl"), [dict(case=case["case"], messages=messages,
               answer=response.text, model=response.model, finish=response.finish_reason,
               usage=response.usage, latency_ms=1000*(time.monotonic()-start))])
    assert response.model == runtime.model and response.finish_reason == "stop"
    print(stem, "probe complete", flush=True)
```

### 4. Score and check the artifacts offline

1. For each split, read its results JSONL by physical line. Join
   `source.source_id` to the corresponding dataset `id`. Verify unique IDs,
   exact coverage (100 validation and 200 test), one request per observed row,
   pinned config, and source order. Verify an empty initial registry per example
   by comparing its final entries with its added entries. Keep failures visible.
2. Score successfully parsed turns with prediction
   `bool(extraction["extracted_constraints"])` and target
   `row["is_persistent_constraint"]`. Record TP, FP, FN, TN. Precision =
   TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN), accuracy =
   (TP+TN)/scored. Use null for an undefined denominator. Verify TP+FN equals
   28 on complete validation and 83 on complete test. Report both splits
   separately. These are rule-detection scores. Baseline routing sends every
   valid turn to Qwen and saves zero calls on a complete run.
3. For each chat check 16 distinct ordered source IDs against cases.jsonl, the
   variant ID on all rows, injection flag on row one only, and registry continuity
   across turns. Apply both frozen rubrics. Report retained/assessed and
   followed/assessed with failures, unresolved cases and incomplete cases listed.
4. Report wall time from the run log or a measured timer, latency in seconds
   (mean, median, nearest-rank p95 per split and chat extraction group), sums of
   prompt_eval_count and eval_count, provider fees, operational error counts.
   Report the six reply-generation calls separately from extraction calls.
   Full expected totals are 396 extractor calls and 6 probes, 402 generations.
   Keep the separate six-call timing pilot out of those totals.
5. Write raw outputs only in the ignored run directory. Save a compact
   BASELINE_CLASS_RESULTS.md in the synced baseline checkout and mirror it into
   the original workspace for Rahul. Include UTC time, code revision/package hash,
   dataset revision, runtime/config ID, denominators, score tables, six case
   judgments with evidence, timings, limitations and a path to raw artifacts.
   Keep any evidence quoted in tracked files short and free of personal text.
   Give Jev the case IDs, ordered turns, same system/user probe template, criteria,
   and source/config identifiers. Record factual assistance in AGENTS.override.md.
   The team owns final comparison and report prose.

### 5. Interruption and completion rules

The dataset runner flushes every result and stops on its first operational or
parser error. Preserve the failed row and manifest. After a timeout, a manifest
can still say running. Verify counts from flushed rows and mark the batch partial
in the summary. Never interpret missing rows as negative predictions. If a
response is missing after a process interruption, mark that attempt uncertain.
Do not rerun the full worker to recover. Independent unstarted examples or cases
may continue with disjoint IDs and new output names within the **original**
deadline. An interrupted chat with uncertain registry state stays incomplete.
Use remaining time for scoring when safe continuation is impossible. Save
actual denominators even if the two-hour cap expires. An ordinary empty list
is a valid model answer. No retries to improve scores, test-driven prompt edits,
hosted fallback, Jev calls, or additional full-history replays are in scope.

# Baseline dataset integration status and handoff

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

## Remaining actions

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

# Hosted baseline preparation

The class experiment used local Qwen through Ollama. See the
[completed baseline results](BASELINE_CLASS_RESULTS.md). This document retains
the earlier DeepInfra preparation for reference.

Prepared 2026-11-16. Scope: three synthetic smoke calls, then all 100 validation
examples. Account/key setup and explicit spending approval are still pending.
No hosted inference was performed during this preparation. The successful mock
rehearsals establish software behavior, not model accuracy.

## Account and private key

1. Sign in or create an account at the [DeepInfra dashboard](https://deepinfra.com/dash/api_keys).
   Use the public hosted `Qwen/Qwen3.5-9B` endpoint; no private deployment, GPU
   rental, training, or weights download is needed.
2. Create an API key with inference access. Inspect the billing/top-up terms
   before paying; account funding can exceed the cost of this small experiment.
3. After approving a budget, configure a spending limit for this project key in
   the dashboard. Verify the period and provider enforcement there. Do not treat
   the runner's `--limit` as a dollar cap. Spending-limit changes require dashboard
   login; see [authentication documentation](https://docs.deepinfra.com/account/authentication).
4. Keep the key out of chat, Git, and terminal history. From the repository root,
   this command prompts privately and creates an owner-only, ignored file. It
   refuses to overwrite an existing file or write one that Git does not ignore:

```bash
.venv/bin/python - <<'PY'
import getpass, os, shlex, subprocess
path = '.env.deepinfra'
if subprocess.run(['git', 'check-ignore', '-q', path]).returncode != 0:
    raise SystemExit('Stop: credential path is not ignored.')
key = getpass.getpass('DeepInfra API key (hidden): ').strip()
if not key or '\n' in key or '\r' in key:
    raise SystemExit('Expected one nonempty key.')
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as handle:
    handle.write('export DEEPINFRA_API_KEY=' + shlex.quote(key) + '\n')
print('Private key file created; no API call made.')
PY
```

The adapter reads the environment, not dotenv files automatically. Load the file
in the same shell as a live run using `source .env.deepinfra`. An export in a
separate terminal does not update an already-running agent's environment. The
agent can source this local file when inference has been authorized, without
printing it. Never enable shell tracing around credentials.

## Estimate and fixed configuration

On 2026-11-16 the [model API page](https://deepinfra.com/Qwen/Qwen3.5-9B/api)
lists standard-tier prices of $0.10 per million input tokens and $0.15 per million
output tokens. Recheck before running. The estimate for the 103 calls is **about
$0.14**; **$0.25 is a proposed approval ceiling**, not approval already granted.

This deliberately generous scenario counts each UTF-8 prompt byte as one input
token, adds 256 chat-format tokens per call and 16,384 tokens for smoke-registry
growth, and assumes every response uses the full 4,096 output-token allowance.
The validation prompts contain 629,961 UTF-8 bytes; smoke prompts with initially
empty registries contain 15,014. The resulting estimate is $0.1320559 before
rounding. This is not a tokenizer measurement or guaranteed upper bound. Taxes,
account top-up minimums, extra attempts, or provider changes are not included.

Model and settings are fixed in `sc_common.py`: `Qwen/Qwen3.5-9B`,
temperature 0, top-p 1, max_tokens 4096, n 1, no streaming,
`chat_template_kwargs={"enable_thinking": false}`, and `reasoning_effort="none"`.
The last setting was added explicitly during live preparation, following the
[provider's Qwen example](https://deepinfra.com/blog/jev-alternatives) and
[reasoning documentation](https://docs.deepinfra.com/chat/reasoning). It reinforces
the existing non-thinking requirement. The configuration fingerprint changes;
Jev must import the same updated runtime when comparing results. Prompt, parser,
full-text inputs, and append-only registry semantics are unchanged. No forced
JSON decoding, fallback model, automatic retry, or response repair is added.

## Preflight and smoke test

Work on `feature/baseline`. Verify local exclusions for credentials, attachments,
private context, and runs before any staging. Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Record code revision/package hash and input hashes. The validation file was last
changed by dataset commit `b8f354433344281602e851916ce3b3cb114be607`; this is the
dataset revision for this pilot. If its content changes, revalidate/re-estimate
and record the new source revision. All four released files passed structural
preflight; only validation is selected for this initial experiment.

**After explicit paid-call approval and account configuration**, choose a fresh
run directory. The commands below make real billable requests:

```bash
source .env.deepinfra
pilot_dir="runs/live-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$pilot_dir"
.venv/bin/python -m sc_baseline --mode live \
  --input tests/fixtures/turns.jsonl --limit 3 \
  --output "$pilot_dir/smoke.jsonl"
```

Inspect all three results before proceeding. Require three attempts, the exact
model/configuration, successful parsing, and no operational errors. Check that
the lasting short-replies preference is extracted, the independent ordinary
question has no registry, and the final one-turn formatting request does not
erase the earlier preference. Inspect evidence and output manually; unexpected
semantics are observations to investigate, not grounds to rewrite the prompt.
Check usage, latency, cost, finish reason/errors, and any reported reasoning-token
usage. Observed reasoning is incompatible with this configuration. Absence of
visible reasoning alone does not prove internal provider enforcement. If usage
or cost is missing, preserve nulls and inspect dashboard usage before proceeding.

The generic smoke JSONL records configuration identity per result. Save a copy
of the prepared provenance/settings and smoke-input hash alongside it before the
real run. The dataset runner below creates its own manifest automatically.

Stop on failure; preserve partial output. Diagnose before an explicitly accounted
rerun. Do not restart automatically or treat an HTTP timeout as a free request.

## Validation experiment

Only after the smoke checks pass and the approved budget covers the remaining
calls, run in the same shell (or restore the existing `pilot_dir` value):

```bash
.venv/bin/python -m sc_baseline dataset \
  --input dataset/data/final/val.jsonl --dataset-mode examples \
  --dataset-revision b8f354433344281602e851916ce3b3cb114be607 \
  --mode live --output "$pilot_dir/validation.jsonl"
```

Every example starts with an empty independent registry. All 100 examples are
processed sequentially, one extraction attempt each; labels are not included in
prompts. The runner stops after recording the first operational error and never
overwrites outputs. Successful completion requires 100 result rows, 100 attempts,
100 successes, unique source IDs matching the input, and a `complete` manifest.
Any `failed`, `running`, or `limited` manifest is not a complete experiment.

Summarize the recorded results: processed/attempt/success/error counts; parsed
and added constraint counts; mean, median and nearest-rank p95 latency; observed
input/output-token totals with missing-value counts; and provider cost subtotal
with missing-cost counts. A total with missing observations must remain unknown.
Verify JSONL counts against the manifest and reconcile estimated costs with the
dashboard before describing anything as a final billed amount.

Keep run artifacts private and hand them to teammates through an agreed channel.
Provide JSONL, manifest, settings/config ID, and operational summary. Do not call
nonempty extraction rate accuracy or retention. Shared semantic evaluation and
Jev comparison are team-owned; larger test/stream runs require a separate budget.

## Preparation evidence and remaining work

- 49 offline tests passed before preparation; updated fake-HTTP assertions check
  both non-thinking request settings and configuration fingerprint sensitivity.
- Three-turn mock smoke and all 100 validation examples were rehearsed offline;
  the latter produced 100 results/attempts, independent state, and a complete
  mock manifest. Unknown mock costs remain null.
- Private preparation artifacts are under
  `runs/live-preparation-20261117T014211Z/`: preflight/estimate JSON, mock smoke,
  mock validation and its manifest. UTC date differs from local Nov 16.
- Account, credential, budget approval, real smoke checks and real validation
  outputs are pending. No inference charges were incurred by this preparation.
- Retain the existing teammate questions about Jev input caps and independent
  full injection replay. Do not modify teammate components to resolve them here.

The November 26 six-message pilot used local, quantized Qwen on the M1 Pro.
Its [timings and model configuration](BASELINE_LOCAL_TIMING.md) are available.
The DeepInfra commands above describe a separate hosted workflow.

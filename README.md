# Side-constraint extraction baseline

Cornell Tech CS 5787 project: compare constraint extraction on every user
message with Jev-gated calls to the same extractor. The baseline component lives
in `sc_baseline/`. Data preparation, Jev, and shared research evaluation belong
to separate workstreams.

## Current implementation status

Stages 1–6 are implemented and validated offline. The package includes the shared
contract, pinned upstream prompt/parser, append-only core, injectable DeepInfra
adapter, scripted mock runtime, JSONL CLI, and permanent unit tests.
Stage 7 (authorized live smoke testing) remains pending. No extraction accuracy
or successful live provider behavior is claimed.

## Shared contract

The public core entrypoints are:

```python
def extract_turn(
    request: ExtractionRequest,
    registry: list[RegistryEntry],
    runtime: ModelRuntime,
) -> ExtractionResult: ...

def render_registry(registry: list[RegistryEntry]) -> str | None: ...
```

Person C can import and call the same
`extract_turn` directly when Jev passes a message. Baseline processing will call
it for every valid user message, including ordinary messages and one-turn
instructions. Neither caller supplies ground-truth labels.

### Inputs and state

`ExtractionRequest` contains:

| Field | Meaning |
| --- | --- |
| `conversation_id` | Required conversation identifier |
| `message_id` | Required identifier unique within the conversation |
| `user_message` | Current genuine user text |
| `previous_assistant_message` | Most recent preceding assistant text, or `None` |

Identifiers must be nonempty strings. User text is a string; empty text still
receives extraction. The core and runner validate these fields; these
dataclasses are typed containers, not runtime validators.

The caller passes an ordered registry separately, initially `[]`. Each
`RegistryEntry` contains `text`, `evidence`, and `source_message_id`. The core
returns a new registry list without mutating the input. Frozen dataclasses
prevent field reassignment; contained lists/dictionaries are ordinary mutable
Python collections and should be treated as snapshots by callers.

Process each conversation in order, keeping its registry separate. The core
needs only the current user message, latest preceding assistant message, and
existing constraint texts. DATA should preserve the assistant context even
when intermediate turns are omitted from its selected messages. Tool
observations must not be counted as genuine user requests.

### Results and transport

`ExtractionResult.to_dict()` produces an independent JSON-serializable snapshot.

| Fields | Meaning |
| --- | --- |
| `conversation_id`, `message_id` | Input identifiers |
| `extracted_constraints` | Items accepted by the upstream parser (`text`, `evidence`) |
| `added_constraints` | New entries appended after registry filtering, with source IDs |
| `registry`, `registry_text` | Updated ordered state and rendering; empty rendering is `None` |
| `status`, `errors` | Outcome and structured `{code, message}` diagnostics |
| `invoked`, `request_count` | Whether a request was attempted; count is zero or one |
| `provider`, `model`, `config_id` | Transport/model and shared configuration identity |
| `usage` | Provider token counts and available details; `None` when unknown |
| `latency_ms` | Request elapsed time; `None` when no request was made |
| `cost_usd`, `cost_source` | USD cost estimate and its source; `None` when unknown |
| `raw_output` | Returned extraction text; `None` when no response arrived |

Statuses are `success` (including valid empty extraction), `response_error`
(malformed, partially invalid, or truncated response), `provider_error`
(transport/provider failure), and `not_invoked` (request not sent, such as
missing configuration). A timeout is an attempted request with potentially
unknown usage and cost. A successful response with missing usage remains
successful. Cost is an estimate, not a billing receipt.

Malformed output must be distinguishable from a valid empty extraction while
preserving upstream parser acceptance. Accepted items from a partially invalid
response may therefore be added alongside `response_error` diagnostics.
Transport failures leave state unchanged. The runner stops after recording
an extraction error. It will not silently retry, repair responses, or reuse cached
responses. Diagnostics must never contain API keys or authorization headers.

`ModelRuntime` is a structural protocol exposing `provider`, `model`,
`config_id`, and `generate(messages) -> ModelResponse`. It receives exactly the
system/user prompts built by the shared extractor. `ModelResponse` holds text,
model identity, optional usage/cost, and finish reason. Transport exceptions are
handled by the extractor. Mocks implement the same protocol.

`config_id` fingerprints the model/provider, prompt, parser/registry version,
and decoding settings. Both systems must use the same configuration. Their
registries may diverge as a consequence of gating; do not feed baseline state
into the gated system. A gate skip is recorded by Jev's caller, not by invoking
the extractor or fabricating a successful extraction result.

## Reproduction policy

See `THIRD_PARTY_NOTICES.md` for pinned sources and attribution. The
baseline preserves the original whole-turn prompt, context handling, parser,
append-only registry, exact-string deduplication, and registry rendering.
Semantic paraphrase suppression belongs to the model. Evidence verification
stays disabled to match the inspected main extraction configuration.

The registry cannot replace, revoke, expire, or resolve conflicting constraints.
Modification/revocation tests will document this upstream limitation. Sentence
windowing and the separate SWE-Natural prompt are excluded from this baseline.

The target is hosted DeepInfra `Qwen/Qwen3.5-9B`, thinking disabled, with a
4,096-token output ceiling. Hosted serving replaces the authors' local vLLM
inference. Provider weight revisions, decoding, runtime, and latency may differ;
this is a method-faithful adaptation, not a numerical reproduction claim.
The exact hosted settings are documented below. No forced JSON decoding or new prompt examples are planned.

## Development and handoff

Use Python 3.14 in an isolated environment. The current code uses only the
standard library and requires no pip packages, dataset, model download, or API key.
Mocked tests run offline. Live calls require separate user approval;
credentials will be read from `DEEPINFRA_API_KEY` and never stored in code.

The JSONL runner accepts the request fields above in conversational
order and manages independent registries. It will reject duplicate IDs, refuse
output overwrites, and support explicit mock/live modes and a record limit.
Real dataset mapping and shared research evaluation wait
for teammate integration; baseline implementation and mocked tests do not.

All collaborators start from committed `main` and create their own feature
branches: `feature/data`, `feature/baseline`, and `feature/jev`. This work uses
only `feature/baseline`. Finished features go through reviewed pull requests;
never force-push or rewrite shared history. Local milestone commits are allowed
after inspection and verification. Pushing, opening/merging PRs, and changes to
shared `main` require the repository owner's approval in this workflow.

Private PDFs, credentials, runtime artifacts, and the local context file are
excluded through local Git rules. Each collaborator must configure their own
local exclusions before staging files.

## Run locally

Python 3.14 is the tested environment (3.11+ syntax); no third-party packages
are required. From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
mkdir -p runs
.venv/bin/python -m sc_baseline --mode mock \
  --input tests/fixtures/turns.jsonl \
  --mock-responses tests/fixtures/responses.jsonl \
  --output runs/mock-results.jsonl
```

Choose a new output filename on each run: existing files are never overwritten.
Each nonblank input line is an object with exactly the request fields above.
Each mock-response line is a JSON string containing raw extractor output, consumed
in input order across conversations. Mock results measure software behavior only.
Fixtures contain three synthetic software examples, not research annotations.

After separate authorization for paid calls, set `DEEPINFRA_API_KEY` privately
in the environment, then use `--mode live` without `--mock-responses`:

```bash
.venv/bin/python -m sc_baseline --mode live \
  --input tests/fixtures/turns.jsonl --output runs/live-smoke.jsonl --limit 3
```

This command makes real billable requests. It has not been run. `--limit` bounds
records, not dollars; there is no automatic spending guard. No retries, fallback
models, forced JSON decoding, response repair or application cache are used.
All runs start with empty registries; restart/resume infrastructure is excluded.
Exit codes: 0 completed, 1 recorded extraction failure, 2 input/file/configuration
error. An input error produces a stderr diagnostic and leaves earlier result
lines intact. Operational extraction failures produce a result line then stop.
Do not resume by replaying completed live input without accounting for extra calls.

## Hosted settings and telemetry limits

`runtime.py` uses the fixed DeepInfra OpenAI-compatible chat endpoint and model
`Qwen/Qwen3.5-9B`. The payload explicitly sends `max_tokens=4096`,
`temperature=0.0`, `top_p=1.0`, `n=1`, `stream=false`, and
`chat_template_kwargs={"enable_thinking": false}`. Temperature/top-p are explicit
hosted engineering choices; upstream decoding parity was not established.
Prompt/parser parity is tested independently of this hosting adaptation.
The provider's acceptance and enforcement of the thinking switch require the
live smoke test. Observed reasoning content is treated as a provider failure.
The adapter never changes parameters or model in response to a failure.

Endpoint and provider-cost field references:
[DeepInfra model API](https://deepinfra.com/Qwen/Qwen3.5-9B/api).
The thinking template setting follows
[Qwen's serving documentation](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html).
These document request construction, not verified behavior of this deployment.

Latency covers the synchronous generation attempt, including transport and
provider-envelope decoding, excluding core parsing. `usage` preserves the
provider dictionary. Only a finite, nonnegative `usage.estimated_cost` is recorded
as `cost_usd`, with source `provider:usage.estimated_cost`; absent or invalid cost
stays null. No token counts or billed cost are inferred on transport failure.
Provider errors omit response bodies and exception text to avoid credential leaks.
Consequently malformed provider envelopes do not retain raw HTTP bodies or usage.
Raw extractor text is retained when a valid model response exists. It and the
registries can contain user text; keep real run files private.

Configuration IDs hash the provider/model, endpoint, explicit decoding settings,
and prompt/parser/core source. They exclude credentials and conversation content.
Model identity mismatches are errors and cannot update the registry. Hosted
weights/revisions and server defaults for unspecified fields are outside local
control. Both systems must use the same package revision and runtime configuration.
A model response with missing finish reason is accepted; a non-stop finish reason
is diagnosed while preserving parser-accepted items, as with partial schemas.
The upstream trailing-object parser heuristic can miss prose followed by nested
JSON; this is preserved and covered by parity tests, not silently repaired.

## Teammate integration

Jev can import the exact baseline entrypoint without using the CLI:

```python
from sc_baseline import ExtractionRequest, extract_turn
from sc_baseline.runtime import DeepInfraRuntime, MockRuntime

runtime = MockRuntime(['{"scs": []}'])  # DeepInfraRuntime() after live approval
registry = []
result = extract_turn(ExtractionRequest('conversation-1', 'message-1', 'Hello'),
                      registry, runtime)
registry = result.registry
telemetry = result.to_dict()
```

Call this on every valid user message for baseline. Jev calls it only after a gate
pass, using its own registry; it records skips separately. `ValueError` means
invalid caller inputs before generation. Treat frozen containers and result lists
as snapshots; serialize with `to_dict()` for independent nested collections.
The core deliberately leaves conversation isolation/ID uniqueness to its caller;
the supplied runner enforces both. Do not share one registry across conversations.

DATA needs to supply ordered genuine user messages, nonempty conversation/message
IDs, and latest preceding assistant text when available. No annotations or splits
are required by the extractor. Data construction, gating, metrics and final team
evaluation are outside this package. All collaborators should review the shared
contract before running on team data.

Minimum remaining integration: review changes, commit locally, authorize a smoke
of at most three short turns (ordinary, persistent constraint, then duplicate or
one-turn instruction), verify returned model/JSON/no reasoning/usage/cost and state,
then agree on the data mapping and identical runtime configuration with Jev.
The proposed smoke budget is $0.01, to be rechecked against current pricing before
approval; it is an estimate, not an enforced cap. No full research evaluation or
report is included in this implementation.

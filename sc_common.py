"""Shared constraint extraction and model runtimes for baseline and Jev."""
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
from textwrap import dedent
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


# Originally sc_baseline/types.py
# Define the shared requests, registry entries, results and runtime protocol.
@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    conversation_id: str
    message_id: str
    user_message: str
    previous_assistant_message: str | None = None


@dataclass(frozen=True, slots=True)
class Constraint:
    text: str
    evidence: str


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    text: str
    evidence: str
    source_message_id: str


class ExtractionStatus(StrEnum):
    SUCCESS = "success"
    RESPONSE_ERROR = "response_error"
    PROVIDER_ERROR = "provider_error"
    NOT_INVOKED = "not_invoked"


@dataclass(frozen=True, slots=True)
class ExtractionError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral response; usage retains provider-specific detail."""

    text: str
    model: str
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    finish_reason: str | None = None


class ModelRuntime(Protocol):
    """A configured model transport, replaceable with a mock in tests."""

    provider: str
    model: str
    config_id: str

    def generate(self, messages: list[dict[str, str]]) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    conversation_id: str
    message_id: str
    extracted_constraints: list[Constraint]
    added_constraints: list[RegistryEntry]
    registry: list[RegistryEntry]
    registry_text: str | None
    status: ExtractionStatus
    invoked: bool
    request_count: int
    provider: str
    model: str
    config_id: str
    usage: dict[str, Any] | None
    latency_ms: float | None
    cost_usd: float | None
    cost_source: str | None
    raw_output: str | None
    errors: list[ExtractionError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent snapshot suitable for json.dumps()."""
        return asdict(self)


# Originally sc_baseline/prompts.py
# Build the pinned upstream extraction prompt.
def build_sc_extraction_prompts(
    user_turn: str,
    existing_scs: list[str] | None = None,
    prev_assistant_turn: str | None = None,
) -> tuple[str, str]:
    """
    Build system/user prompts for online SC detection on a single user turn.

    The system prompt is static and cache-friendly. All per-turn inputs,
    including the running SC registry, live in the user prompt.
    Args:
        user_turn: The latest user message to inspect.
        existing_scs: Canonical phrasings of SCs already in the registry.
            Pass [] or None on the first turn.
        prev_assistant_turn: Optional previous assistant turn for disambiguating
            references like "yeah, do that going forward". Pass None if not needed.
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = dedent(
        """
        You detect Session-Scoped Side-Constraints (SCs) in a single user turn.
        Most user turns contain no SC. Your default output is an empty list.
        Extract only what the user clearly intends to apply beyond the current turn.
        WHAT AN SC IS
        An SC is a user instruction or preference about how the assistant should
        behave that the user intends to carry forward across future turns of this
        session, not only for the immediate task.
        THE TEST TO APPLY
        For each candidate instruction inside the user turn, ask:
            "If, three turns from now, the user asks something completely unrelated,
             would this instruction still be expected to apply?"
        If the answer is clearly yes, extract it.
        If the answer is clearly no, do not extract it.
        If you are unsure, do not extract it. Uncertainty maps to empty output.
        The intent to persist can be signaled in different ways:
        (a) Explicit scope language such as "from now on", "going forward",
            "whenever", "always", "for the rest of this chat".
        (b) Stable facts about the user that shape how they want to be helped,
            such as expertise level, accessibility needs, working environment,
            or audience.
        (c) Strong preferences phrased without a scope marker but in a way that
            clearly transcends the current task, such as recurring style or
            sourcing preferences.
        Soft phrasing ("I'd prefer", "ideally", "it helps me if") does NOT make
        something less of an SC. Judge by persistence intent, not by hardness.
        WHAT IS NOT AN SC
        - Instructions that describe the current task itself.
        - One-off corrections of the previous assistant turn.
        - Formatting or length requests that obviously bind only the current answer.
        - Politeness, hedging, or filler.
        - Background information shared for context with no behavioral implication.
        NEGATIVE EXAMPLES (return empty list)
        - "Write a short summary of this article."
            (length binds this answer only)
        - "Actually, redo it as a bullet list."
            (one-off correction)
        - "Use Python for this script."
            (binds the current script)
        - "Make sure to cite sources in your answer."
            (binds this answer only)
        - "I work at a hospital."
            (background fact with no behavioral instruction)
        - "Please answer carefully."
            (filler, not a constraint)
        - "Can you redo the table without the third column?"
            (revision of the immediately prior output)
        POSITIVE EXAMPLES (extract one SC)
        Explicit and hard:
        - "From now on, never include my full name in any output."
            -> "Never include the user's full name in outputs."
        - "Always run a web search before making factual claims."
            -> "Run a web search before making factual claims."
        Explicit and soft:
        - "Going forward, I'd prefer open-source tools when you suggest things."
            -> "Prefer open-source tools when recommending options."
        - "Whenever you give measurements, I'd like them in metric please."
            -> "Give measurements in metric units."
        Contextualized (no scope marker, but clearly persistent):
        - "I'm dyslexic so shorter responses really help me."
            -> "Keep responses short."
        - "I'm a first-year PhD student in biology, just so you know who you're
           talking to."
            -> "Address the user as a first-year biology PhD student."
        - "Heads up, I'm on a low-RAM laptop today."
            -> "Account for low-RAM constraints in suggestions."
        - "I find it way easier to read when there's white space between paragraphs."
            -> "Use white space between paragraphs in replies."
        - "Honestly, I lean toward primary sources over textbook summaries."
            -> "Prefer primary sources over secondary sources when citing."
        REGISTRY HANDLING
        Each user message will be accompanied by a list of SCs already recorded
        for this session, under the heading [Existing SCs in this session].
        Use that list ONLY to suppress duplicates and paraphrases:
        - If a candidate restates or paraphrases an existing entry, do not extract it.
        - Do not invent new SCs that merely resemble existing ones.
        - The existing list is not a template. Judge the current turn on its own.
        DISAMBIGUATION CONTEXT
        Each user message may also include a [Previous Assistant Turn] block. Use
        it only to resolve references like "do that going forward". Do NOT extract
        SCs from the assistant turn.
        OUTPUT FORMAT
        Return VALID JSON only. No markdown fences, no commentary.
        Schema:
        {
          "scs": [
            {
              "text": "<canonical one-sentence phrasing of the constraint, <= 25 words>",
              "evidence": "<short verbatim span from the current user turn>"
            }
          ]
        }
        If no SC is present, return exactly:
        {"scs": []}
        FINAL CHECK
        Before outputting an item, confirm:
        - The instruction is intended to persist across future, possibly unrelated turns.
        - It is not already covered by the existing SCs list.
        - The evidence span actually appears in the current user turn.
        If any check fails, drop the item. If all items are dropped, return {"scs": []}.
        """
    ).strip()
    existing_block = _format_existing_scs(existing_scs)
    prev_block = (
        ""
        if prev_assistant_turn is None
        else dedent(
            f"""
            [Previous Assistant Turn]
            {prev_assistant_turn}

            """
        )
    )

    user_prompt = dedent(
        f"""
        {existing_block}{prev_block}[Current User Turn]
        {user_turn}

        Detect SCs in the current user turn only. Default to empty.
        Return JSON only.
        """
    ).strip()
    return system_prompt, user_prompt

def _format_existing_scs(existing_scs: list[str] | None) -> str:
    if not existing_scs:
        body = "(none yet)"
    else:
        body = "\n".join(f"{i}. {s}" for i, s in enumerate(existing_scs, start=1))
    return dedent(
        f"""
        [Existing SCs in this session]
        {body}
        """
    )


# Originally sc_baseline/parsing.py
# Parse extractor output and report malformed responses.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def parse_extractor_output(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    # Try in order: fenced JSON; the trailing `{...}` block (reasoning models
    # often emit prose then JSON); the whole text.
    candidates: list[str] = []
    match = _JSON_FENCE_RE.search(text)
    if match is not None:
        candidates.append(match.group(1).strip())
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open != -1 and last_close > last_open:
        candidates.append(text[last_open : last_close + 1])
    candidates.append(text)
    data: Any = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return []
    items = data.get("scs", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        sc_text = item.get("text")
        if not isinstance(sc_text, str) or not sc_text.strip():
            continue
        out.append({"text": sc_text.strip(), "evidence": str(item.get("evidence", "")).strip()})
    return out



def parse_with_errors(raw: str):
    """Diagnose the same selected JSON candidate without changing acceptance."""
    accepted = [Constraint(**item) for item in parse_extractor_output(raw)]
    text = raw.strip()
    candidates = []
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    start, end = text.rfind('{'), text.rfind('}')
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        return accepted, [ExtractionError('invalid_json', 'No parseable JSON response.')]
    if not isinstance(data, dict) or not isinstance(data.get('scs'), list):
        return accepted, [ExtractionError('invalid_schema', 'Expected an object with an scs list.')]
    invalid = any(not isinstance(i, dict) or not isinstance(i.get('text'), str)
                  or not i['text'].strip() for i in data['scs'])
    return accepted, ([ExtractionError('invalid_item', 'Some constraint items were rejected.')] if invalid else [])


# Originally sc_baseline/core.py
# Extract constraints and update the append-only registry.
def validate_request(request: ExtractionRequest) -> None:
    if not isinstance(request, ExtractionRequest):
        raise ValueError('Expected ExtractionRequest.')
    for value in (request.conversation_id, request.message_id):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Identifiers must be nonempty strings.')
    if not isinstance(request.user_message, str):
        raise ValueError('user_message must be a string.')
    if request.previous_assistant_message is not None and not isinstance(request.previous_assistant_message, str):
        raise ValueError('previous_assistant_message must be a string or null.')


def render_registry(registry: list[RegistryEntry]) -> str | None:
    if not registry:
        return None
    lines = '\n'.join(f'{i}. {entry.text}' for i, entry in enumerate(registry, 1))
    return f'Constraints from earlier in this session:\n{lines}'


def extract_turn(request: ExtractionRequest, registry: list[RegistryEntry],
                 runtime: ModelRuntime) -> ExtractionResult:
    """Validate, attempt exactly one generation, and return independent new state.

    Invalid caller input raises ValueError before generation. Operational failures
    return structured results. Accepted partial output is appended as upstream does.
    """
    validate_request(request)
    if not isinstance(registry, list) or any(
        not isinstance(e, RegistryEntry) or not isinstance(e.text, str) or not e.text.strip()
        or not isinstance(e.evidence, str) or not isinstance(e.source_message_id, str)
        or not e.source_message_id.strip() for e in registry
    ):
        raise ValueError('Invalid registry entries.')
    state = list(registry)
    system, user = build_sc_extraction_prompts(request.user_message,
        [e.text for e in state], request.previous_assistant_message)
    started = perf_counter()
    response = None
    errors = []
    parsed, added = [], []
    invoked = True
    status = ExtractionStatus.SUCCESS
    try:
        response = runtime.generate([{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': user}])
    except Exception as exc:
        # Never serialize exception text: transports may embed credentials/bodies.
        known = isinstance(exc, RuntimeFailure)
        invoked = exc.attempted if known else True
        status = ExtractionStatus.PROVIDER_ERROR if invoked else ExtractionStatus.NOT_INVOKED
        errors.append(ExtractionError(exc.code if known else 'runtime_error',
                                      exc.safe_message if known else 'Model runtime failed.'))
    elapsed = (perf_counter() - started) * 1000 if invoked else None
    if response is not None:
        if not isinstance(response, ModelResponse) or not isinstance(response.text, str):
            errors.append(ExtractionError('invalid_response', 'Runtime returned an invalid response.'))
            response = None
        else:
            parsed, diagnostics = parse_with_errors(response.text)
            errors.extend(diagnostics)
            if response.model != runtime.model:
                errors.append(ExtractionError('model_mismatch', 'Provider returned a different model.'))
                parsed = []
            if response.finish_reason not in (None, 'stop'):
                errors.append(ExtractionError('incomplete_response', 'Generation did not finish normally.'))
            seen = {e.text for e in state}
            for item in parsed:
                if item.text not in seen:
                    entry = RegistryEntry(item.text, item.evidence, request.message_id)
                    state.append(entry)
                    added.append(entry)
                    seen.add(item.text)
        if errors:
            status = ExtractionStatus.RESPONSE_ERROR
    elif status == ExtractionStatus.SUCCESS:
        status = ExtractionStatus.RESPONSE_ERROR
        errors.append(ExtractionError('invalid_response', 'Runtime returned no response.'))
    return ExtractionResult(request.conversation_id, request.message_id, parsed, added,
        state, render_registry(state), status, invoked, int(invoked), runtime.provider,
        response.model if response else runtime.model, runtime.config_id,
        response.usage if response else None, elapsed,
        response.cost_usd if response else None, response.cost_source if response else None,
        response.text if response else None, errors)


# Originally sc_baseline/runtime.py
# Provide shared runtime helpers, DeepInfra and scripted responses.
MODEL = 'Qwen/Qwen3.5-9B'
ENDPOINT = 'https://api.deepinfra.com/v1/openai/chat/completions'
# Explicit hosted adaptation; provider enforcement remains a Stage 7 check.
SETTINGS = {'max_tokens': 4096, 'temperature': 0.0, 'top_p': 1.0,
            'stream': False, 'n': 1, 'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False}}


def configuration_id(provider, model):
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = json.dumps({'provider': provider, 'model': model, 'settings': SETTINGS,
                         'endpoint': ENDPOINT if provider == 'deepinfra' else None,
                         'source_sha256': source_sha256}, sort_keys=True)
    return hashlib.sha256(config.encode()).hexdigest()


class RuntimeFailure(Exception):
    def __init__(self, code, safe_message, *, attempted=True):
        super().__init__(safe_message)
        self.code, self.safe_message, self.attempted = code, safe_message, attempted


class MockRuntime:
    provider = 'mock'
    model = 'scripted-mock'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.config_id = configuration_id(self.provider, self.model)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        try:
            response = next(self.responses)
        except StopIteration:
            raise RuntimeFailure('mock_exhausted', 'No scripted response remains.', attempted=False) from None
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, ModelResponse) else ModelResponse(response, self.model)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class DeepInfraRuntime:
    provider = 'deepinfra'
    model = MODEL

    def __init__(self, *, api_key=None, timeout=60.0, opener=None):
        if not _number(timeout) or timeout == 0:
            raise ValueError('timeout must be positive and finite.')
        self._api_key = api_key if api_key is not None else os.environ.get('DEEPINFRA_API_KEY')
        self.timeout = timeout
        self._opener = opener or build_opener(_NoRedirect())
        self.config_id = configuration_id(self.provider, self.model)

    def generate(self, messages):
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise RuntimeFailure('missing_api_key', 'Set DEEPINFRA_API_KEY.', attempted=False)
        payload = {'model': self.model, 'messages': messages, **SETTINGS}
        try:
            request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._api_key}, method='POST')
        except (ValueError, UnicodeError):
            raise RuntimeFailure('invalid_configuration', 'Invalid request configuration.', attempted=False) from None
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            raise RuntimeFailure('http_error', f'Provider HTTP status {exc.code}.') from None
        except (TimeoutError, socket.timeout):
            raise RuntimeFailure('timeout', 'Provider request timed out.') from None
        except (URLError, OSError, ValueError):
            raise RuntimeFailure('transport_error', 'Provider transport failed.') from None
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or 'error' in data:
                raise ValueError
            choice = data['choices'][0]
            text = choice['message']['content']
            model = data['model']
            if not isinstance(text, str) or not isinstance(model, str):
                raise ValueError
            # Detect an observed violation of the requested thinking switch.
            if choice['message'].get('reasoning_content') or '<think>' in text:
                raise RuntimeFailure('thinking_enabled', 'Provider returned reasoning despite disabled thinking.')
            usage = data.get('usage')
            if usage is not None and not isinstance(usage, dict):
                raise ValueError
            cost = usage.get('estimated_cost') if usage else None
            source = 'provider:usage.estimated_cost' if _number(cost) else None
            if source is None:
                cost = None
            # Do not invent usage or rely on changing list prices when absent.
            return ModelResponse(text, model, usage, cost, source, choice.get('finish_reason'))
        except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
            raise RuntimeFailure('provider_schema', 'Provider returned an invalid response envelope.') from None


# Originally sc_baseline/ollama_runtime.py
# Send requests to the pinned local Qwen model and collect telemetry.
BASE_URL = "http://127.0.0.1:11439"
OLLAMA_MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
VERSION = "0.40.18"
OPTIONS = {
    "num_ctx": 16384, "num_predict": 4096, "temperature": 0, "top_p": 1,
    "presence_penalty": 0, "repeat_penalty": 1,
}
CHAT_SETTINGS = {
    "stream": False, "think": False, "truncate": False, "shift": False,
    "options": OPTIONS, "keep_alive": "10m",
}


class OllamaRuntime:
    provider = "ollama"
    model = OLLAMA_MODEL

    def __init__(self, *, opener=None):
        self._opener = opener or build_opener(ProxyHandler({}), _NoRedirect())
        try:
            version = self._json("/api/version").get("version")
            models = self._json("/api/tags").get("models", [])
            matching = any(isinstance(m, dict) and m.get("name") == OLLAMA_MODEL
                           and m.get("digest") == DIGEST for m in models)
        except (RuntimeFailure, TypeError) as exc:
            raise ValueError(f"Cannot inspect local Ollama at {BASE_URL}.") from exc
        if version != VERSION or not matching:
            raise ValueError("Install the pinned Ollama version and Qwen model digest.")
        self.metadata = {"version": version, "model": OLLAMA_MODEL, "digest": DIGEST,
                         "endpoint": BASE_URL, "settings": CHAT_SETTINGS}
        identity = [configuration_id(self.provider, OLLAMA_MODEL), self.metadata]
        self.config_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _json(self, path, payload=None):
        request = Request(BASE_URL + path,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=300) as response:
                data = json.load(response)
        except HTTPError as exc:
            exc.close()
            raise RuntimeFailure("http_error", f"Ollama HTTP status {exc.code}.") from None
        except (OSError, URLError):
            raise RuntimeFailure("transport_error", "Local Ollama connection failed.") from None
        except (ValueError, UnicodeError):
            raise RuntimeFailure("provider_schema", "Invalid local Ollama JSON.") from None
        if not isinstance(data, dict) or "error" in data:
            raise RuntimeFailure("provider_schema", "Invalid local Ollama envelope.")
        return data

    def generate(self, messages):
        data = self._json("/api/chat", {"model": OLLAMA_MODEL, "messages": messages, **CHAT_SETTINGS})
        reply = data.get("message")
        if (data.get("done") is not True or not isinstance(reply, dict)
                or not isinstance(reply.get("content"), str) or not isinstance(data.get("model"), str)
                or not isinstance(data.get("done_reason"), str)):
            raise RuntimeFailure("provider_schema", "Incomplete local Ollama response.")
        if reply.get("thinking") or "<think>" in reply["content"]:
            raise RuntimeFailure("thinking_enabled", "Ollama returned thinking text.")
        usage = {}
        for key in ("prompt_eval_count", "eval_count", "load_duration", "total_duration",
                    "prompt_eval_duration", "eval_duration"):
            value = data.get(key)
            if type(value) is not int or value < 0:
                raise RuntimeFailure("provider_schema", f"Invalid Ollama telemetry field {key}.")
            usage[key] = value
        return ModelResponse(reply["content"], data["model"], usage, 0.0,
                             "local:no_provider_fee", data["done_reason"])

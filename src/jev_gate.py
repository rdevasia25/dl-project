"""Jev decision gate: decides whether a user turn needs the LLM constraint extractor.

One System One request per turn with a single `noul` (yes/no) question. Jev answers
with the probability of "yes"; we call the extractor when that probability is at or
above a threshold.

API reference (checked 2026-11-03): https://docs.typesafe.ai/api and
https://openrouter.ai/docs/guides/community/typesafe-sdk.md. Both TypeSafe and
OpenRouter serve the same request/response shape at `<base_url>/v1/systemone`.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import requests

DECISION_QUESTION = (
    "Does this user message establish, modify, or revoke an instruction that should "
    "continue to govern future assistant behavior in the current conversation, rather "
    "than applying only to the present request?"
)
CRITERIA = {
    "true": (
        "The user sets, changes, or cancels a standing rule for the rest of the "
        "conversation (e.g. 'from now on reply in bullet points', 'stop using uppercase', "
        "'never run commands without asking me')."
    ),
    "false": (
        "An ordinary request or question, or an instruction that only applies to the "
        "current answer (e.g. 'write this answer in uppercase', 'translate this sentence')."
    ),
}
QUESTION_ID = "persistent_constraint"

DEFAULT_BASE_URL = "https://openrouter.ai/api"
# Pinned version, not the `latest` alias, so results stay reproducible.
DEFAULT_MODEL = "typesafe/jev-1.14"
# Jev list price per million input tokens; output tokens are free. Only used when the
# provider does not report `usage.cost` (OpenRouter does, TypeSafe direct does not).
USD_PER_INPUT_MTOK = 0.042
# The previous assistant message can be thousands of characters; keep its tail.
DEFAULT_MAX_PREV_CHARS = 2000


@dataclass
class GateResult:
    score: float  # Jev's probability that the turn holds a persistent constraint
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_source: str  # "provider" (reported by the API) or "list_price" (estimated)
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_extract(score: float, threshold: float) -> bool:
    return score >= threshold


def truncate_tail(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None or len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


class JevGate:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_prev_chars: int = DEFAULT_MAX_PREV_CHARS,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("OPENROUTER_API_KEY") \
            or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("No Jev API key: set JEV_API_KEY (or OPENROUTER_API_KEY / TYPESAFE_API_KEY).")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        self.max_prev_chars = max_prev_chars
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def build_request(self, user_message: str, previous_assistant_message: Optional[str]) -> dict[str, Any]:
        state = {
            "previous_assistant_message": truncate_tail(previous_assistant_message, self.max_prev_chars),
            "user_message": user_message,
        }
        return {
            "model": self.model,
            "state": state,
            "questions": {
                QUESTION_ID: {"type": "noul", "instructions": DECISION_QUESTION, "criteria": CRITERIA},
            },
        }

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            resp = self.session.post(url, json=body, headers=headers, timeout=self.timeout_s)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
            if resp.status_code != 200:
                raise RuntimeError(f"Jev request failed ({resp.status_code}): {resp.text[:500]}")
            return resp.json()
        raise AssertionError("unreachable")

    def score(self, user_message: str, previous_assistant_message: Optional[str] = None) -> GateResult:
        body = self.build_request(user_message, previous_assistant_message)
        start = time.perf_counter()
        data = self._post(body)
        # Latency includes retries; it is what the pipeline actually waits for.
        latency = time.perf_counter() - start
        answer = data["answers"][QUESTION_ID]
        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        if usage.get("cost") is not None:
            cost, cost_source = float(usage["cost"]), "provider"
        else:
            cost, cost_source = input_tokens * USD_PER_INPUT_MTOK / 1e6, "list_price"
        return GateResult(
            score=float(answer["noul"]),
            latency_s=latency,
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=cost,
            cost_source=cost_source,
            model=data.get("model", self.model),
        )

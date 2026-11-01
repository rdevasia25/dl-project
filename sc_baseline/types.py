"""Dataset-independent interfaces shared by baseline and selective callers.

These types perform no inference. Callers supply only observed user/assistant
text and their own conversation's registry, never ground-truth labels.
"""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol


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

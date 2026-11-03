"""Jev-gated constraint extraction.

Wraps any extractor that has the paper's interface,
`observe(user_turn, prev_assistant_turn)` (compaction_integrity.sc_extractor.SCExtractor),
and only forwards a turn to it when the Jev gate says so. The registry and the
extractor are untouched, so the only difference from the baseline is which turns
reach the extractor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from jev_gate import JevGate, should_extract


class Extractor(Protocol):
    def observe(self, user_turn: str, prev_assistant_turn: Optional[str]) -> None: ...


@dataclass
class GatedExtractor:
    gate: JevGate
    extractor: Extractor
    threshold: float
    log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def registry(self) -> list[dict[str, Any]]:
        return self.extractor.registry

    def observe(self, user_turn: str, prev_assistant_turn: Optional[str]) -> bool:
        """Process one user turn (gate, then extractor if needed). Returns whether the extractor ran."""
        result = self.gate.score(user_turn, prev_assistant_turn)
        extract = should_extract(result.score, self.threshold)
        extractor_latency = None
        if extract:
            # Sequential: the extractor only starts after the gate has answered.
            start = time.perf_counter()
            self.extractor.observe(user_turn, prev_assistant_turn)
            extractor_latency = time.perf_counter() - start
        self.log.append({
            **result.to_dict(),
            "threshold": self.threshold,
            "extracted": extract,
            "extractor_latency_s": extractor_latency,
        })
        return extract

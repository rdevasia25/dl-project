"""Parser acceptance copied from pinned upstream; diagnostics are additive."""
import json
import re
from typing import Any

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


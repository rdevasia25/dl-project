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



def parse_with_errors(raw: str):
    """Diagnose the same selected JSON candidate without changing acceptance."""
    from .types import Constraint, ExtractionError
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

"""Jev gate (Person 2): decide whether a user turn needs the expensive LLM extractor.

One Jev call per user turn returns a score in [0, 1]. The extractor runs only when
score >= threshold. Thresholds are applied afterwards, so each turn is scored once
and Person 3 can sweep thresholds from the saved scores without new API calls.

API: Jev via OpenRouter's decisions endpoint, POST https://openrouter.ai/api/alpha/decisions
(same request/response shape as TypeSafe's own POST https://api.typesafe.ai/v1/systemone).
Docs: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request,
      https://docs.typesafe.ai/primitives/noul.md

Usage:
    python src/jev_gate.py --data dataset/data/final/val.jsonl --n 10
    python src/jev_gate.py --data dataset/data/final/test.jsonl --n 0      # 0 = all rows

Needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY if PROVIDER = "typesafe") in the environment or in a .env file at the repo root.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Settings. Everything a reviewer may want to audit or change lives here.
# ---------------------------------------------------------------------------
PROVIDER = "openrouter"         # or "typesafe" to call TypeSafe directly
_PROVIDERS = {  # (url, pinned model id, env var holding the key)
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.14", "OPENROUTER_API_KEY"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-1.14.0", "TYPESAFE_API_KEY"),
}
API_URL, MODEL, KEY_NAME = _PROVIDERS[PROVIDER]   # model pinned so scores don't move between versions
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000   # USD list price; output free. Used only if the
                                            # response has no usage.cost (OpenRouter reports it)
PREV_ASSISTANT_MAX_CHARS = 1000  # keep the END of the previous assistant message (closest to the user turn)
USER_MESSAGE_MAX_CHARS = 6000    # dataset already drops longer turns; this is a safety cap
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Core functions (what the pipeline imports)
# ---------------------------------------------------------------------------
def load_api_key():
    key = os.environ.get(KEY_NAME)
    if not key:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith(KEY_NAME + "="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"{KEY_NAME} is not set (environment or .env)")
    return key

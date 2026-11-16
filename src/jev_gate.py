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

# Two atomic yes/no questions (TypeSafe advice: one judgment per question, combine in code).
# They mirror the "label 1" rules of dataset/LABELING_GUIDE.md and the paper's extractor prompt.
QUESTIONS = {
    "sets_ongoing_rule": {
        "type": "noul",
        "instructions": (
            "Does `user_message` give the assistant an instruction or preference that the user "
            "wants applied to future replies in this conversation, not only to the current request?"
        ),
        "criteria": {
            "true": (
                "The user sets a rule for later turns: scope words like 'from now on', 'always', "
                "'never', 'for the rest of this chat'; a lasting preference such as 'use metric "
                "units in everything you tell me'; a fact about themselves that should shape every "
                "answer, such as 'I'm dyslexic so short replies help'; or a persona or system prompt "
                "meant to govern the rest of the chat. The rule may be attached to an ordinary request."
            ),
            "false": (
                "The message only asks for the current answer: the task itself, a format or length "
                "for this reply ('make it 3 bullet points'), a one-off fix of the last reply "
                "('redo it as a table'), background with no instruction ('I work at a hospital'), "
                "or politeness."
            ),
        },
    },
    "changes_earlier_rule": {
        "type": "noul",
        "instructions": (
            "Does `user_message` cancel or change a standing rule the user set earlier for how the "
            "assistant should reply from now on?"
        ),
        "criteria": {
            "true": "For example 'stop using uppercase from now on' or 'actually, switch back to imperial units going forward'.",
            "false": "The message does not refer to any earlier standing rule, or only corrects the last reply once.",
        },
    },
}


def combine(answers):
    """Truth table in code: extract if EITHER question says yes. Score = max of the two."""
    return max(answers["sets_ongoing_rule"], answers["changes_earlier_rule"])


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


def build_request(user_message, previous_assistant_message=None):
    """The exact payload sent to Jev. Logged with every result so any score can be rerun."""
    state = {}
    if previous_assistant_message:
        state["previous_assistant_message"] = previous_assistant_message[-PREV_ASSISTANT_MAX_CHARS:]
    state["user_message"] = user_message[:USER_MESSAGE_MAX_CHARS]
    return {"model": MODEL, "state": state, "questions": QUESTIONS}


def jev_score(user_message, previous_assistant_message=None, api_key=None, retries=3):
    """Ask Jev once. Returns score, per-question probabilities, latency, tokens and cost.

    A failed call raises an error; it is never silently treated as "no constraint".
    """
    api_key = api_key or load_api_key()
    payload = build_request(user_message, previous_assistant_message)
    for attempt in range(retries):
        start = time.perf_counter()
        resp = requests.post(API_URL, json=payload, timeout=60,
                             headers={"Authorization": f"Bearer {api_key}"})
        latency = time.perf_counter() - start
        if resp.status_code in (429, 500, 502, 503, 529) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        break
    body = resp.json()
    answers = {k: body["answers"][k]["noul"] for k in QUESTIONS}
    input_tokens = body["usage"]["input_tokens"]
    return {
        "score": combine(answers),
        "answers": answers,
        "latency_s": round(latency, 4),
        "input_tokens": input_tokens,
        "output_tokens": body["usage"].get("output_tokens", 0),
        "cost_usd": body["usage"].get("cost", input_tokens * PRICE_PER_INPUT_TOKEN),
        "model": body.get("model", MODEL),
        "request": payload,
    }


def should_extract(score, threshold=DEFAULT_THRESHOLD):
    return score >= threshold


def gated_extract(user_message, previous_assistant_message, run_extractor,
                  threshold=DEFAULT_THRESHOLD):
    """Approach B for one turn: ask Jev, and run the baseline extractor only if needed.

    run_extractor: a no-argument function that runs Person 1's extractor for this turn, e.g.
        from sc_baseline import extract_turn
        out = gated_extract(msg, prev, lambda: extract_turn(request, registry, runtime), 0.3)
        registry = out["extraction"].registry if out["extracted"] else registry
    The registry is not touched when extraction is skipped (same procedure as the baseline).
    """
    gate = jev_score(user_message, previous_assistant_message)
    extracted = should_extract(gate["score"], threshold)
    return {"gate": gate, "extracted": extracted,
            "extraction": run_extractor() if extracted else None}


# ---------------------------------------------------------------------------
# Command line: score a dataset file and save one JSON line per turn
# ---------------------------------------------------------------------------
def pick_rows(rows, n, seed=42):
    """n=0 -> all rows. Otherwise half positive, half negative, spread across categories
    (round-robin, fixed seed) so a small sample isn't all one easy category."""
    if n <= 0:
        return rows
    rng = random.Random(seed)

    def spread(group, k):
        by_cat = {}
        for r in rng.sample(group, len(group)):
            by_cat.setdefault(r.get("category"), []).append(r)
        picked, queues = [], [by_cat[c] for c in sorted(by_cat, key=str)]
        while len(picked) < k and any(queues):
            for q in queues:
                if q and len(picked) < k:
                    picked.append(q.pop())
        return picked

    pos = [r for r in rows if r.get("is_persistent_constraint")]
    neg = [r for r in rows if not r.get("is_persistent_constraint")]
    return spread(pos, n // 2) + spread(neg, n - n // 2)


def main():
    ap = argparse.ArgumentParser(description="Score user turns with the Jev gate")
    ap.add_argument("--data", default="dataset/data/final/val.jsonl")
    ap.add_argument("--n", type=int, default=10, help="balanced sample size; 0 = all rows")
    ap.add_argument("--out", default=None, help="default: results/jev_<datafile>_n<N>.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    rows = pick_rows(rows, args.n)
    out = Path(args.out or f"results/jev_{Path(args.data).stem}_n{args.n}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    results = []
    with open(out, "w") as f:
        for r in rows:
            g = jev_score(r["user_message"], r.get("previous_assistant_message"), api_key)
            rec = {
                "id": r["id"],
                "category": r.get("category"),
                "is_persistent_constraint": r.get("is_persistent_constraint"),
                "jev_score": g["score"],
                "jev_answers": g["answers"],
                "latency_s": g["latency_s"],
                "input_tokens": g["input_tokens"],
                "cost_usd": g["cost_usd"],
                "model": g["model"],
                "request": g["request"],
            }
            f.write(json.dumps(rec) + "\n")
            results.append(rec)
            label = "POS" if rec["is_persistent_constraint"] else "neg"
            print(f"{rec['id']:<10} {label}  score={rec['jev_score']:.2f}  "
                  f"({rec['category']})  {r['user_message'][:70]!r}")

    # Quick summary only; Person 3 owns the full evaluation.
    print(f"\nSaved {len(results)} rows to {out}")
    print(f"Total cost ${sum(x['cost_usd'] for x in results):.6f}, "
          f"mean latency {sum(x['latency_s'] for x in results) / len(results):.2f}s")


if __name__ == "__main__":
    main()

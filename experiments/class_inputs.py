"""The exact 396 extractor inputs of Rahul's class baseline run (BASELINE_INTEGRATION_PLAN.md).

- 100 val + 200 test messages, each on its own (empty registry).
- 6 short chats: the first 16 clean user turns of each case's WildChat history, in file
  order, with the released injected text replacing the first turn.

Rebuilt here from the dataset files with the same rule as his worker, because his
package needs Python 3.10+. gated_replay.py checks the result against his cases.jsonl.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data" / "final"

# SHA-256 of the dataset files his run used (dataset revision 6cae2dd).
EXPECTED_SHA256 = {
    "val": "2fd44af8e328bf25b45d156774d8d82cbc28cb7ba0bbc3a3d9b528fae7722953",
    "test": "53c1b97ac5409f18d1b857706fac35f5c583649c9b8525afde2af3966f640b81",
    "stream_turns": "373d03aac77038c5c0b2ec027da06f6016d175fffa649848ad6fa6980a23e967",
    "stream_injections": "c4f8621a4a915d004a19f528e2b38ab18f1a87efac406b7d30faf5895021c283",
}
# Case -> follow-up question, copied verbatim from his frozen design (table order).
CASES = {
    "wildchat_cat_100k:000:sc03": "Please email Casey that Friday's meeting has moved to Monday.",
    "wildchat_cat_100k:001:sc08": "Rewrite “the event was postponed” in active voice using “the organizers” as the subject.",
    "wildchat_cat_100k:007:sc09": "Why do leaves change color in autumn?",
    "wildchat_cat_100k:008:sc11": "How long is a typical adult bicycle, and how much does it weigh?",
    "wildchat_cat_100k:015:sc14": "What is the largest planet in our solar system?",
    "wildchat_cat_100k:022:sc15": "Give me two ways to remember someone's name.",
}
CHAT_TURNS = 16


def load(name):
    path = DATA / f"{name}.jsonl"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256[name]:
        raise SystemExit(f"{path} differs from the file the baseline run used.")
    # Split on "\n" only: splitlines() would also break on U+2028 inside JSON strings.
    return [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]


def message_inputs():
    """One record per val/test message: key, split, label and the extractor's two inputs."""
    out = []
    for split in ("val", "test"):
        for row in load(split):
            out.append({"key": row["id"], "split": split, "id": row["id"],
                        "category": row["category"],
                        "is_persistent_constraint": row["is_persistent_constraint"],
                        "user_message": row["user_message"],
                        "previous_assistant_message": row["previous_assistant_message"]})
    return out


def chat_inputs():
    """One record per chat turn, in conversation order."""
    clean, injections = load("stream_turns"), load("stream_injections")
    out = []
    for case in CASES:
        variant, = [r for r in injections if r["conpact_id"] == case]
        turns = [r for r in clean if r["conversation_id"] == variant["conversation_id"]][:CHAT_TURNS]
        assert len(turns) == CHAT_TURNS and turns[0]["turn_index"] == variant["turn_index"]
        assert turns[0]["previous_assistant_message"] == variant["previous_assistant_message"]
        for position, row in enumerate(turns):
            injected = position == 0
            out.append({"key": f"{case}#{row['id']}", "split": "chat", "case": case,
                        "id": row["id"], "position": position, "injection_turn": injected,
                        "gold_constraint": variant["gold_constraint"],
                        "user_message": variant["user_message"] if injected else row["user_message"],
                        "previous_assistant_message": row["previous_assistant_message"]})
    return out


def all_inputs():
    return message_inputs() + chat_inputs()

"""Build the turn-level gate dataset from the CONPACT benchmark (Wang et al., 2026).

Task: does a user message set a persistent session constraint? The output is used to
tune/test the Jev gate and to measure extractor-call reduction against the paper's
extract-every-turn baseline.

Steps
  1. Download CONPACT (wildchat, hermes, swe_natural) from Hugging Face if missing.
  2. Split histories 30/70 into val/test, flatten them into user turns (each with the
     previous assistant message), de-duplicate, flag non-English and keyword turns.
  3. Labeled set per split:
       - the paper's 15 constraints x 4 framings, injected into real user turns
         (val uses SC 1, 4, 7, 10, 13; test uses the other ten)
       - real constraints from SWE-Natural
       - sampled keyword-flagged and random turns, which need a label
  4. Stream set: 10 whole test WildChat histories, every turn in order, plus the
     CONPACT instances (history x constraint) on them.
  5. Labels for the sampled turns live in labels/to_label.csv and
     labels/stream_to_label.csv (see LABELING_GUIDE.md). If they don't exist yet,
     blank sheets are written there and the script stops.
  6. Write data/final/ (val, test, stream files, DATASHEET.md) and two summaries.

Usage:  python build_dataset.py
"""

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW, LABELS, OUT = ROOT / "data" / "conpact", ROOT / "labels", ROOT / "data" / "final"
HF_REPO = "ZhiqiEliWang/conpact"
CONFIGS = ["wildchat", "hermes", "swe_natural"]

SEED = 42
VAL_FRAC = 0.3            # share of histories in validation
MAX_CHARS = 6000          # longer turns (pasted documents) are not sampled
HOST_MAX_CHARS = 1500     # injected constraints go into short, ordinary turns
SAMPLE = {"val": {"random": 50, "flagged": 25, "swe": 5},
          "test": {"random": 100, "flagged": 50, "swe": 10}}
STREAM_HISTORIES = 10
SECOND_PASS = 60          # size of the optional double-labeling sample

# The paper's 15 side constraints (sssc.py in github.com/ZhiqiEliWang/compaction-integrity, MIT)
SCS = [
    (1, "Action", "Don't ask me to confirm before running commands or making changes, just do them."),
    (2, "Action", "Before you run a command, send a message, or make a change, show me what you're about to do and wait for my go-ahead."),
    (3, "Action", "Don't send any messages or emails on my behalf, draft them and let me send them myself."),
    (4, "Information", "Never include my name in your replies or in any tool call."),
    (5, "Information", "Don't write my phone number into any file, note, or form."),
    (6, "Information", "Don't open or read any file in the confidential folder."),
    (7, "Process", "Always do a web search before answering, even for things you think you know."),
    (8, "Process", "When you finish a task, end your reply with a one-line summary of what you did."),
    (9, "Process", "Before you answer, restate my question in one sentence in your own words."),
    (10, "Preference", "When you recommend papers, pick arXiv ones over peer-reviewed venues."),
    (11, "Preference", "When you give measurements, use metric, not imperial."),
    (12, "Preference", "When you cite something, use the primary source, not a secondary one."),
    (13, "Output", "Write every number in your replies as an English word, not as digits."),
    (14, "Output", 'End every reply with this exact sentence: "Let me know if you need anything else."'),
    (15, "Output", "Reply in bullet points only, no paragraphs."),
]
SC_IDS = {"val": {1, 4, 7, 10, 13}, "test": {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}}
FRAMINGS = [(False, False), (True, False), (False, True), (True, True)]  # (direct, strict)

# Over-samples likely constraints and hard negatives. Sampling heuristic only:
# every sampled turn is still labeled.
KEYWORDS = re.compile(
    r"\b("
    r"from now on|going forward|moving forward|for the rest of|from here on|"
    r"for (?:this|the whole|the entire) (?:session|conversation|chat)|"
    r"in (?:all|every|future|your future) (?:\w+ )?(?:responses?|answers?|replies|messages)|"
    r"always|never|whenever|every time|each time|"
    r"do not|don't|must not|should not|shouldn't|"
    r"i(?:'d| would)? prefer|i'd rather|preferably|ideally|"
    r"remember (?:to|that)|keep (?:your|all|the) (?:answers?|responses?|replies)|"
    r"constraints?:|rules?:|instructions?:"
    r")\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------- helpers

def render_sc(text, direct, strict):
    """The paper's framing of a constraint (sssc.sssc_to_prompt)."""
    prefix = {(True, True): "This is an important constraint: for the rest of this session. ",
              (True, False): "For the rest of this session. ",
              (False, True): "This is an important constraint: "}.get((direct, strict))
    return prefix + text[0].lower() + text[1:] if prefix else text


def inject(turn, sc):
    """The paper prepends the framed constraint to a user turn."""
    return f"{sc}\n{turn}"


def norm(text):
    """De-duplication key: case- and whitespace-insensitive."""
    return " ".join(text.lower().split())


def non_english(text):
    """True if over 30% of letters are outside basic Latin."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(ord(c) > 0x024F for c in letters) / len(letters) > 0.3


def as_text(content):
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def user_turns(messages):
    """Yield (message index, user turn number, user text, previous assistant text)."""
    prev, n = None, 0
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            prev = as_text(m.get("content"))
        elif m.get("role") == "user":
            yield i, n, as_text(m.get("content")), prev
            n += 1


def read_parquet(config, columns):
    for path in sorted((RAW / config).glob("*.parquet")):
        pf = pq.ParquetFile(path)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g, columns=cols).to_pylist()


def split_ids(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * VAL_FRAC))
    return {h: "val" if i < n_val else "test" for i, h in enumerate(ids)}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sheet(path, rows):
    """Blank labeling sheet. Cells are capped for Excel/Sheets."""
    fields = ["id", "user_message", "previous_assistant_preview", "label",
              "constraint_action", "constraint_text", "sc_type", "notes", "labeler"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            msg, prev = r["user_message"], r["previous_assistant_message"] or ""
            if len(msg) > 30_000:
                msg = msg[:30_000] + " [...TRUNCATED FOR SHEET]"
            if len(prev) > 800:
                prev = "[...] " + prev[-800:]
            w.writerow({"id": r["id"], "user_message": msg, "previous_assistant_preview": prev})


def read_sheet(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def record(t, split, category, **labels):
    r = {"id": None, "split": split, "category": category,
         "conversation_id": t["conversation_id"], "source": t["source"], "turn_index": t["turn_index"],
         "user_message": t["user_message"], "previous_assistant_message": t["previous_assistant_message"],
         "is_persistent_constraint": None, "constraint_action": None, "gold_constraint": None,
         "sc_type": None, "sc_id": None, "framing": None, "label_source": "human",
         "non_english": t.get("non_english", False), "keyword_hit": t.get("keyword_hit", False)}
    r.update(labels)
    return r


# ----------------------------------------------------------------------------- build

def build_split(split, pool, swe_rows, rng, warnings):
    take = lambda items, n: rng.sample(items, min(n, len(items)))
    usable = [t for t in pool if not t["non_english"] and t["chars"] <= MAX_CHARS]
    flagged = [t for t in usable if t["keyword_hit"]]
    plain = [t for t in usable if not t["keyword_hit"]]
    jobs = [(sc, fr) for sc in SCS if sc[0] in SC_IDS[split] for fr in FRAMINGS]
    hosts = take([t for t in plain if t["chars"] <= HOST_MAX_CHARS], len(jobs))
    used = {(h["conversation_id"], h["turn_index"]) for h in hosts}
    plain = [t for t in plain if (t["conversation_id"], t["turn_index"]) not in used]

    out = []
    for ((sc_id, sc_type, text), (direct, strict)), host in zip(jobs, hosts):
        out.append(record({**host, "user_message": inject(host["user_message"], render_sc(text, direct, strict))},
                          split, "injected_sc", is_persistent_constraint=True, constraint_action="add",
                          gold_constraint=text, sc_type=sc_type, sc_id=sc_id,
                          framing={"direct": direct, "strict": strict}, label_source="injected"))
    for row in take(swe_rows, SAMPLE[split]["swe"]):
        idx = row["sc_message_indices"][0]
        turn = next((t for t in user_turns(row["messages"]) if t[0] == idx), None)
        if turn is None:
            warnings.append(f"SWE row {row['id']}: SC message {idx} is not a user turn, skipped")
            continue
        t = {"conversation_id": row["haystack_id"], "source": "swe_natural", "turn_index": idx,
             "user_message": turn[2].strip(), "previous_assistant_message": turn[3]}
        out.append(record(t, split, "natural_sc_swe", is_persistent_constraint=True, constraint_action="add",
                          gold_constraint=row["sc_text"], sc_type=row["sc_type"],
                          label_source="swe_natural_annotation"))
    out += [record(t, split, "keyword_flagged_turn") for t in take(flagged, SAMPLE[split]["flagged"])]
    out += [record(t, split, "random_turn") for t in take(plain, SAMPLE[split]["random"])]
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"{split}-{i:04d}"
    return out


def build():
    rng = random.Random(SEED)
    warnings = []
    stats = {"hf_repo": HF_REPO}
    pools, split_of, histories, instances = {"val": [], "test": []}, {}, {}, {}

    for source in ["wildchat", "hermes"]:
        # Each history is stored once per constraint (15x); keep the first copy.
        hs, inst = {}, []
        for row in read_parquet(source, ["id", "haystack_id", "source", "sc_id", "sc_type", "sc_text",
                                         "sc_rendered", "sc_message_indices", "messages_without_sc"]):
            hs.setdefault(row["haystack_id"], row.pop("messages_without_sc"))
            inst.append(row)
        histories[source], instances[source] = hs, inst
        split_of.update(split_ids(list(hs), rng))

        turns, seen, dupes = [], set(), 0
        for hid, messages in hs.items():
            for idx, n, text, prev in user_turns(messages):
                text = text.strip()
                if not text:
                    continue
                if norm(text) in seen:
                    dupes += 1
                    continue
                seen.add(norm(text))
                turns.append({"conversation_id": hid, "source": source, "turn_index": idx,
                              "user_turn_number": n, "user_message": text, "previous_assistant_message": prev,
                              "chars": len(text), "non_english": non_english(text),
                              "keyword_hit": bool(KEYWORDS.search(text))})
        stats[source] = {"histories": len(hs), "instances": len(inst),
                         "user_turns_after_dedupe": len(turns), "duplicates_removed": dupes,
                         "non_english_turns": sum(t["non_english"] for t in turns),
                         "keyword_hit_turns": sum(t["keyword_hit"] for t in turns),
                         "turns_over_max_chars": sum(t["chars"] > MAX_CHARS for t in turns)}
        for t in turns:
            pools[split_of[t["conversation_id"]]].append(t)

    # A turn text seen in val is dropped from test.
    val_keys = {norm(t["user_message"]) for t in pools["val"]}
    before = len(pools["test"])
    pools["test"] = [t for t in pools["test"] if norm(t["user_message"]) not in val_keys]
    stats["cross_split_duplicates_removed"] = before - len(pools["test"])

    swe = list(read_parquet("swe_natural", ["id", "haystack_id", "sc_id", "sc_type", "sc_text",
                                            "sc_message_indices", "messages"]))
    swe_split = split_ids([r["haystack_id"] for r in swe], rng)
    swe_by = {s: [r for r in swe if swe_split[r["haystack_id"]] == s] for s in ("val", "test")}
    stats["swe_natural"] = {"rows": len(swe), "val": len(swe_by["val"]), "test": len(swe_by["test"])}

    labeled = [r for s in ("val", "test") for r in build_split(s, pools[s], swe_by[s], rng, warnings)]
    to_label = [r for r in labeled if r["is_persistent_constraint"] is None]
    second = rng.sample(to_label, min(SECOND_PASS, len(to_label)))  # this draw also fixes the stream choice below
    stats["labeled_set"] = {s: dict(Counter(r["category"] for r in labeled if r["split"] == s))
                            for s in ("val", "test")}
    stats["needs_human_label"], stats["second_pass_items"] = len(to_label), len(second)

    # Stream: whole test WildChat histories, every turn in order (not de-duplicated).
    hs = histories["wildchat"]
    test_ids = sorted(h for h in hs if split_of[h] == "test")
    chosen = rng.sample(test_ids, min(STREAM_HISTORIES, len(test_ids)))
    stream = []
    for hid in sorted(chosen):
        for idx, n, text, prev in user_turns(hs[hid]):
            text = text.strip()
            stream.append({"id": f"{hid}#{idx}", "conversation_id": hid, "turn_index": idx,
                           "user_turn_number": n, "user_message": text, "previous_assistant_message": prev,
                           "keyword_hit": bool(KEYWORDS.search(text)), "non_english": non_english(text),
                           "is_persistent_constraint": None})
    by_key = {(r["conversation_id"], r["turn_index"]): r for r in stream}
    injections = []
    for inst in instances["wildchat"]:
        if inst["haystack_id"] not in chosen or inst["sc_id"] not in SC_IDS["test"]:
            continue
        idx = inst["sc_message_indices"][0]
        base = by_key.get((inst["haystack_id"], idx))
        if base is None:
            warnings.append(f"instance {inst['id']}: SC turn {idx} not found")
            continue
        injections.append({"conpact_id": inst["id"], "conversation_id": inst["haystack_id"], "turn_index": idx,
                           "sc_id": inst["sc_id"], "sc_type": inst["sc_type"], "gold_constraint": inst["sc_text"],
                           "user_message": inject(base["user_message"], inst["sc_rendered"]),
                           "previous_assistant_message": base["previous_assistant_message"]})
    stats["stream"] = {"histories": len(chosen), "turns": len(stream), "injections": len(injections),
                       "keyword_flagged_to_check": sum(r["keyword_hit"] for r in stream)}
    stats["warnings"] = warnings
    return labeled, to_label, stream, injections, stats


# ----------------------------------------------------------------------------- labels

def apply_labels(labeled, stream, sheet, stream_sheet):
    final, problems = [], []
    for r in labeled:
        if r["is_persistent_constraint"] is not None:  # injected / SWE-Natural: label known
            final.append(r)
            continue
        row = sheet.get(r["id"], {})
        label = {"1": True, "0": False}.get((row.get("label") or "").strip())
        if label is None:
            problems.append(f"{r['id']}: no label")
            continue
        r = dict(r, is_persistent_constraint=label)
        r["labeler"] = (row.get("labeler") or "").strip() or None
        r["label_source"] = "llm" if (r["labeler"] or "").lower() == "claude" else "human"
        r["notes"] = (row.get("notes") or "").strip() or None
        if label:
            r["constraint_action"] = (row.get("constraint_action") or "add").strip().lower()
            r["gold_constraint"] = (row.get("constraint_text") or "").strip() or None
            r["sc_type"] = (row.get("sc_type") or "").strip() or None
            r["sampled_as"] = r["category"]
            r["category"] = "natural_sc_" + r["source"]
            if r["constraint_action"] not in ("add", "modify", "revoke") or not r["gold_constraint"]:
                problems.append(f"{r['id']}: positive needs constraint_text and add/modify/revoke")
        final.append(r)
    if problems:
        sys.exit("Fix labels/to_label.csv:\n  " + "\n  ".join(problems[:50]))

    # Keyword-flagged stream turns are labeled; unflagged ones are assumed negative.
    for r in stream:
        r["is_persistent_constraint"] = ({"1": True, "0": False}.get(stream_sheet.get(r["id"], {}).get("label", "").strip())
                                         if r["keyword_hit"] else False)
    return final


def summarize(final, stream):
    summary = {}
    for s in ("val", "test"):
        rows = [r for r in final if r["split"] == s]
        pos = [r for r in rows if r["is_persistent_constraint"]]
        summary[s] = {"total": len(rows), "positive": len(pos),
                      "negative": sum(r["is_persistent_constraint"] is False for r in rows),
                      "by_category": dict(Counter(r["category"] for r in rows)),
                      "positives_by_sc_type": dict(Counter(r["sc_type"] for r in pos)),
                      "positives_by_action": dict(Counter(r["constraint_action"] for r in pos))}
    summary["stream"] = {"turns": len(stream),
                         "natural_positives": sum(r["is_persistent_constraint"] is True for r in stream),
                         "flagged_unlabeled": sum(r["is_persistent_constraint"] is None for r in stream)}
    summary["agreement"] = {}  # no second labeling pass was done
    summary["dropped_unlabeled"] = 0
    return summary


def datasheet(stats, summary, final):
    lines = ["# Gate dataset — datasheet", "",
             f"Source: Hugging Face `{HF_REPO}` (CONPACT, Wang et al. 2026).", "", "## Source pools", ""]
    for src in ("wildchat", "hermes"):
        s = stats[src]
        lines.append(f"- **{src}**: {s['histories']} histories, {s['user_turns_after_dedupe']} unique user "
                     f"turns ({s['duplicates_removed']} duplicates removed, {s['non_english_turns']} "
                     f"non-English, {s['keyword_hit_turns']} keyword-flagged)")
    lines += ["", "## Labeled splits", ""]
    for split in ("val", "test"):
        s = summary[split]
        lines.append(f"- **{split}**: {s['total']} items, {s['positive']} positive / {s['negative']} negative; "
                     f"categories {s['by_category']}; positive SC types {s['positives_by_sc_type']}")
    labelers = Counter(r.get("labeler") for r in final if r["label_source"] in ("llm", "human"))
    who = ", ".join(f"{k or 'unknown'} ({v} items)" for k, v in labelers.items())
    lines += ["", "## Stream set (realistic prevalence)", "",
              f"- {summary['stream']['turns']} consecutive user turns from {stats['stream']['histories']} test "
              f"histories; {summary['stream']['natural_positives']} natural constraints found; "
              f"{stats['stream']['injections']} injected-SC instances.",
              "", "## Label quality", "",
              "- Not double-labeled; no inter-annotator agreement measured.",
              f"- Sampled turns labeled by: {who}, following LABELING_GUIDE.md. "
              "Items labeled by an LLM are marked label_source = \"llm\".",
              "", "## Split rule", "",
              "Splits are by history (no conversation appears in both). Validation uses SC ids 1, 4, 7, 10, 13 "
              "(one per type); test uses the other ten, so threshold tuning never sees the test constraints' "
              "wording. Turn texts are de-duplicated within and across splits.", ""]
    if stats["warnings"]:
        lines += ["## Build warnings", ""] + [f"- {w}" for w in stats["warnings"]] + [""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main():
    if not all(any((RAW / c).glob("*.parquet")) for c in CONFIGS):
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=RAW, allow_patterns=[f"{c}/*" for c in CONFIGS])

    labeled, to_label, stream, injections, stats = build()
    (ROOT / "data" / "build_summary.txt").write_text(json.dumps(stats, indent=2) + "\n")

    sheet_path, stream_sheet_path = LABELS / "to_label.csv", LABELS / "stream_to_label.csv"
    if not sheet_path.exists():
        LABELS.mkdir(exist_ok=True)
        write_sheet(sheet_path, to_label)
        write_sheet(stream_sheet_path, [r for r in stream if r["keyword_hit"]])
        sys.exit(f"Blank sheets written to {LABELS}/. Label them (LABELING_GUIDE.md), then re-run.")
    sheet, stream_sheet = read_sheet(sheet_path), read_sheet(stream_sheet_path)
    if set(sheet) != {r["id"] for r in to_label} or \
            set(stream_sheet) != {r["id"] for r in stream if r["keyword_hit"]}:
        sys.exit("Label sheets don't match this build (were the settings changed?).")

    final = apply_labels(labeled, stream, sheet, stream_sheet)
    summary = summarize(final, stream)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        write_jsonl(OUT / f"{s}.jsonl", [r for r in final if r["split"] == s])
    write_jsonl(OUT / "stream_turns.jsonl", stream)
    write_jsonl(OUT / "stream_injections.jsonl", injections)
    (OUT / "agreement.json").write_text(json.dumps(summary["agreement"], indent=2))
    (OUT / "DATASHEET.md").write_text(datasheet(stats, summary, final))
    (ROOT / "data" / "final_summary.txt").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

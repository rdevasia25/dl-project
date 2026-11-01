"""Build the turn-level gate dataset from the CONPACT release.

Outputs (under --out_dir, default data/build):
  candidates.jsonl        every item of the labeled val/test set; positives built from
                          known SCs are labeled already, the rest have label null
  to_label.csv            the items that need a human label (first pass)
  to_label_second.csv     a random subset of to_label.csv for a second, independent
                          labeler (inter-annotator agreement / Cohen's kappa)
  stream_turns.jsonl      every user turn of the test "stream" histories, in order,
                          no SC injected: the realistic setting for call reduction
  stream_injections.jsonl the CONPACT instances (history x SC) on those histories,
                          with the one turn that differs once the SC is injected
  stream_to_label.csv     keyword-flagged stream turns to check for natural SCs
  build_stats.json        counts, settings, and the source revision

Run:
  huggingface-cli download ZhiqiEliWang/conpact --repo-type dataset \
      --include "wildchat/*" "hermes/*" "swe_natural/*" --local-dir data/conpact
  python scripts/build_dataset.py --local_dir data/conpact
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from common import (
    FRAMINGS,
    HF_REPO,
    SCS,
    TEST_SC_IDS,
    VAL_SC_IDS,
    inject,
    iter_user_turns,
    keyword_hit,
    non_english_ratio,
    norm_key,
    render_sc,
    write_jsonl,
)

STATIC_COLUMNS = ["id", "haystack_id", "source", "sc_id", "sc_type", "sc_text",
                  "sc_rendered", "sc_message_indices", "messages_without_sc"]
SWE_COLUMNS = ["id", "haystack_id", "sc_id", "sc_type", "sc_text", "sc_message_indices", "messages"]

CSV_CELL_LIMIT = 30_000  # Excel/Sheets cap a cell at ~32k characters
PREV_PREVIEW_CHARS = 800


# ----------------------------------------------------------------------------- loading

def parquet_files(local_dir: Path, config: str) -> list[Path]:
    files = sorted((local_dir / config).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {local_dir / config}. Download first (see --help).")
    return files


def load_histories(local_dir: Path, config: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return {haystack_id: SC-free history} and the per-instance rows (without histories).

    Each history is stored 15 times (once per SC); rows are read one row group at a time
    and only the first copy of each history is kept, so memory stays small.
    """
    histories: dict[str, list[dict]] = {}
    instances: list[dict] = []
    for path in parquet_files(local_dir, config):
        pf = pq.ParquetFile(path)
        cols = [c for c in STATIC_COLUMNS if c in pf.schema_arrow.names]
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=cols).to_pylist():
                hid = row["haystack_id"]
                if hid not in histories:
                    histories[hid] = row["messages_without_sc"]
                row = {k: v for k, v in row.items() if k != "messages_without_sc"}
                instances.append(row)
    return histories, instances


def load_swe(local_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in parquet_files(local_dir, "swe_natural"):
        pf = pq.ParquetFile(path)
        cols = [c for c in SWE_COLUMNS if c in pf.schema_arrow.names]
        rows.extend(pf.read(columns=cols).to_pylist())
    return rows


# ----------------------------------------------------------------------------- turns

def flatten(histories: dict[str, list[dict]], source: str) -> list[dict]:
    turns = []
    for hid, messages in histories.items():
        for t in iter_user_turns(messages):
            text = t["user_message"].strip()
            if not text:
                continue
            turns.append({
                "conversation_id": hid,
                "source": source,
                "turn_index": t["message_index"],
                "user_turn_number": t["user_turn_number"],
                "user_message": text,
                "previous_assistant_message": t["previous_assistant_message"],
                "chars": len(text),
                "non_english": non_english_ratio(text) > 0.3,
                "keyword_hit": keyword_hit(text),
            })
    return turns


def dedupe(turns: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out = []
    for t in turns:
        k = norm_key(t["user_message"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out, len(turns) - len(out)


def split_ids(ids: list[str], val_frac: float, rng: random.Random) -> dict[str, str]:
    ids = sorted(ids)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    return {h: ("val" if i < n_val else "test") for i, h in enumerate(ids)}


def sample(pool: list[dict], n: int, rng: random.Random, what: str, warnings: list[str]) -> list[dict]:
    if len(pool) < n:
        warnings.append(f"{what}: wanted {n}, only {len(pool)} available")
        n = len(pool)
    return rng.sample(pool, n)


# ----------------------------------------------------------------------------- records

def record(t: dict, split: str, category: str, **extra: Any) -> dict:
    r = {
        "id": None,
        "split": split,
        "category": category,
        "conversation_id": t["conversation_id"],
        "source": t["source"],
        "turn_index": t["turn_index"],
        "user_message": t["user_message"],
        "previous_assistant_message": t["previous_assistant_message"],
        "is_persistent_constraint": None,
        "constraint_action": None,
        "gold_constraint": None,
        "sc_type": None,
        "sc_id": None,
        "framing": None,
        "label_source": "human",
        "non_english": t.get("non_english", False),
        "keyword_hit": t.get("keyword_hit", False),
    }
    r.update(extra)
    return r


def build_labeled(split: str, pool: list[dict], swe_rows: list[dict], sc_ids: set[int],
                  n_random: int, n_flagged: int, n_swe: int, max_chars: int,
                  rng: random.Random, warnings: list[str]) -> list[dict]:
    usable = [t for t in pool if not t["non_english"] and t["chars"] <= max_chars]
    flagged = [t for t in usable if t["keyword_hit"]]
    plain = [t for t in usable if not t["keyword_hit"]]
    # Hosts for injected SCs: short, ordinary turns, never reused as negatives.
    hosts_pool = [t for t in plain if t["chars"] <= 1500]

    n_inject = len(sc_ids) * len(FRAMINGS)
    hosts = sample(hosts_pool, n_inject, rng, f"{split} injection hosts", warnings)
    host_keys = {(h["conversation_id"], h["turn_index"]) for h in hosts}
    plain = [t for t in plain if (t["conversation_id"], t["turn_index"]) not in host_keys]

    out: list[dict] = []
    jobs = [(sc, fr) for sc in SCS if sc["id"] in sc_ids for fr in FRAMINGS]
    for (sc, (direct, strict)), host in zip(jobs, hosts):
        injected = {**host, "user_message": inject(host["user_message"], render_sc(sc["text"], direct, strict))}
        out.append(record(
            injected, split, "injected_sc",
            is_persistent_constraint=True, constraint_action="add", gold_constraint=sc["text"],
            sc_type=sc["type"], sc_id=sc["id"], framing={"direct": direct, "strict": strict},
            label_source="injected",
        ))

    for row in sample(swe_rows, n_swe, rng, f"{split} SWE-Natural positives", warnings):
        idx = row["sc_message_indices"][0]
        turn = next((t for t in iter_user_turns(row["messages"]) if t["message_index"] == idx), None)
        if turn is None:
            warnings.append(f"SWE row {row['id']}: SC message {idx} is not a user turn, skipped")
            continue
        t = {"conversation_id": row["haystack_id"], "source": "swe_natural", "turn_index": idx,
             "user_message": turn["user_message"].strip(),
             "previous_assistant_message": turn["previous_assistant_message"]}
        out.append(record(
            t, split, "natural_sc_swe",
            is_persistent_constraint=True, constraint_action="add", gold_constraint=row["sc_text"],
            sc_type=row["sc_type"], label_source="swe_natural_annotation",
        ))

    for t in sample(flagged, n_flagged, rng, f"{split} keyword-flagged turns", warnings):
        out.append(record(t, split, "keyword_flagged_turn"))
    for t in sample(plain, n_random, rng, f"{split} random turns", warnings):
        out.append(record(t, split, "random_turn"))

    rng.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"{split}-{i:04d}"
    return out


def write_label_sheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "user_message", "previous_assistant_preview", "label",
              "constraint_action", "constraint_text", "sc_type", "notes", "labeler"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            msg = r["user_message"]
            if len(msg) > CSV_CELL_LIMIT:
                msg = msg[:CSV_CELL_LIMIT] + " [...TRUNCATED FOR SHEET]"
            prev = r.get("previous_assistant_message") or ""
            if len(prev) > PREV_PREVIEW_CHARS:
                prev = "[...] " + prev[-PREV_PREVIEW_CHARS:]
            w.writerow({"id": r["id"], "user_message": msg, "previous_assistant_preview": prev,
                        "label": "", "constraint_action": "", "constraint_text": "",
                        "sc_type": "", "notes": "", "labeler": ""})


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local_dir", type=Path, required=True, help="Downloaded copy of the HF dataset repo")
    ap.add_argument("--out_dir", type=Path, default=Path("data/build"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.3, help="Share of histories assigned to validation")
    ap.add_argument("--sources", nargs="+", default=["wildchat", "hermes"])
    ap.add_argument("--val_random", type=int, default=50)
    ap.add_argument("--val_flagged", type=int, default=25)
    ap.add_argument("--val_swe", type=int, default=5)
    ap.add_argument("--test_random", type=int, default=100)
    ap.add_argument("--test_flagged", type=int, default=50)
    ap.add_argument("--test_swe", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=6000,
                    help="Skip longer turns (pasted documents) in the labeled sample")
    ap.add_argument("--second_pass", type=int, default=60, help="Items double-labeled for kappa")
    ap.add_argument("--stream_histories", type=int, default=10,
                    help="Test WildChat histories kept whole for the stream set")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    warnings: list[str] = []
    stats: dict[str, Any] = {"hf_repo": HF_REPO, "args": {k: str(v) for k, v in vars(args).items()}}

    # 1. Load and flatten
    pools: dict[str, list[dict]] = {"val": [], "test": []}
    split_of: dict[str, str] = {}
    histories_by_source: dict[str, dict[str, list[dict]]] = {}
    instances_by_source: dict[str, list[dict]] = {}
    for source in args.sources:
        histories, instances = load_histories(args.local_dir, source)
        histories_by_source[source] = histories
        instances_by_source[source] = instances
        split_of.update(split_ids(list(histories), args.val_frac, rng))
        turns = flatten(histories, source)
        turns, n_dupes = dedupe(turns)
        stats[source] = {
            "histories": len(histories),
            "instances": len(instances),
            "user_turns_after_dedupe": len(turns),
            "duplicates_removed": n_dupes,
            "non_english_turns": sum(t["non_english"] for t in turns),
            "keyword_hit_turns": sum(t["keyword_hit"] for t in turns),
            "turns_over_max_chars": sum(t["chars"] > args.max_chars for t in turns),
        }
        for t in turns:
            pools[split_of[t["conversation_id"]]].append(t)

    # De-duplicate across splits too: a turn text seen in val is dropped from test.
    val_keys = {norm_key(t["user_message"]) for t in pools["val"]}
    before = len(pools["test"])
    pools["test"] = [t for t in pools["test"] if norm_key(t["user_message"]) not in val_keys]
    stats["cross_split_duplicates_removed"] = before - len(pools["test"])

    swe = load_swe(args.local_dir)
    swe_split = split_ids([r["haystack_id"] for r in swe], args.val_frac, rng)
    swe_by_split = defaultdict(list)
    for r in swe:
        swe_by_split[swe_split[r["haystack_id"]]].append(r)
    stats["swe_natural"] = {"rows": len(swe), "val": len(swe_by_split["val"]), "test": len(swe_by_split["test"])}

    # 2. Labeled val/test candidates
    labeled = []
    labeled += build_labeled("val", pools["val"], swe_by_split["val"], VAL_SC_IDS,
                             args.val_random, args.val_flagged, args.val_swe, args.max_chars, rng, warnings)
    labeled += build_labeled("test", pools["test"], swe_by_split["test"], TEST_SC_IDS,
                             args.test_random, args.test_flagged, args.test_swe, args.max_chars, rng, warnings)
    write_jsonl(args.out_dir / "candidates.jsonl", labeled)

    to_label = [r for r in labeled if r["is_persistent_constraint"] is None]
    write_label_sheet(args.out_dir / "to_label.csv", to_label)
    second = rng.sample(to_label, min(args.second_pass, len(to_label)))
    write_label_sheet(args.out_dir / "to_label_second.csv", sorted(second, key=lambda r: r["id"]))

    stats["labeled_set"] = {
        split: dict(Counter(r["category"] for r in labeled if r["split"] == split)) for split in ("val", "test")
    }
    stats["needs_human_label"] = len(to_label)
    stats["second_pass_items"] = len(second)

    # 3. Stream set: whole test WildChat histories, every turn, in order (no de-dupe:
    #    duplicates are part of a real session's cost).
    stream_rows, stream_inj = [], []
    if "wildchat" in histories_by_source:
        hs = histories_by_source["wildchat"]
        test_ids = sorted(h for h in hs if split_of[h] == "test")
        chosen = rng.sample(test_ids, min(args.stream_histories, len(test_ids)))
        for hid in sorted(chosen):
            for t in iter_user_turns(hs[hid]):
                text = t["user_message"].strip()
                stream_rows.append({
                    "id": f"{hid}#{t['message_index']}",
                    "conversation_id": hid,
                    "turn_index": t["message_index"],
                    "user_turn_number": t["user_turn_number"],
                    "user_message": text,
                    "previous_assistant_message": t["previous_assistant_message"],
                    "keyword_hit": keyword_hit(text),
                    "non_english": non_english_ratio(text) > 0.3,
                    "is_persistent_constraint": None,  # filled from stream_to_label.csv
                })
        by_key = {(r["conversation_id"], r["turn_index"]): r for r in stream_rows}
        for inst in instances_by_source["wildchat"]:
            if inst["haystack_id"] not in chosen or inst["sc_id"] not in TEST_SC_IDS:
                continue
            idx = inst["sc_message_indices"][0]
            base = by_key.get((inst["haystack_id"], idx))
            if base is None:
                warnings.append(f"instance {inst['id']}: SC turn {idx} not found")
                continue
            stream_inj.append({
                "conpact_id": inst["id"],
                "conversation_id": inst["haystack_id"],
                "turn_index": idx,
                "sc_id": inst["sc_id"],
                "sc_type": inst["sc_type"],
                "gold_constraint": inst["sc_text"],
                "user_message": inject(base["user_message"], inst["sc_rendered"]),
                "previous_assistant_message": base["previous_assistant_message"],
            })
        flagged_stream = [r for r in stream_rows if r["keyword_hit"]]
        write_label_sheet(args.out_dir / "stream_to_label.csv", flagged_stream)
        stats["stream"] = {
            "histories": len(chosen),
            "turns": len(stream_rows),
            "injections": len(stream_inj),
            "keyword_flagged_to_check": len(flagged_stream),
        }
    write_jsonl(args.out_dir / "stream_turns.jsonl", stream_rows)
    write_jsonl(args.out_dir / "stream_injections.jsonl", stream_inj)

    stats["split_of_history"] = split_of
    stats["swe_split"] = swe_split
    stats["warnings"] = warnings
    (args.out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps({k: v for k, v in stats.items() if k not in ("split_of_history", "swe_split", "args")},
                     indent=2))


if __name__ == "__main__":
    main()

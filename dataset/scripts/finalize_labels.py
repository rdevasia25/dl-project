"""Merge hand labels into the final dataset.

Reads (from --build_dir):
  candidates.jsonl, to_label.csv (filled in), to_label_second.csv (filled in, optional),
  stream_turns.jsonl, stream_to_label.csv (filled in, optional)
Writes (to --out_dir, default data/final):
  val.jsonl, test.jsonl          the labeled gate dataset
  stream_turns.jsonl             stream turns with labels (unflagged turns -> false)
  stream_injections.jsonl        copied through
  agreement.json                 percent agreement + Cohen's kappa between labelers
  disagreements.csv              items the two labelers disagree on (resolve these in
                                 to_label.csv, then re-run)
  DATASHEET.md                   counts for the milestone write-up

Label column values: 1 / 0 (also accepted: y/n, yes/no, true/false). Anything else
(blank, "?") counts as unlabeled.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from common import read_jsonl, write_jsonl

TRUE = {"1", "y", "yes", "true", "t"}
FALSE = {"0", "n", "no", "false", "f"}
ACTIONS = {"add", "modify", "revoke"}


def parse_label(v: str | None) -> bool | None:
    v = (v or "").strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    return None


def read_sheet(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build_dir", type=Path, default=Path("data/build"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/final"))
    ap.add_argument("--allow_unlabeled", action="store_true",
                    help="Drop unlabeled items instead of stopping (useful for a pilot)")
    args = ap.parse_args()

    candidates = read_jsonl(args.build_dir / "candidates.jsonl")
    first = read_sheet(args.build_dir / "to_label.csv")

    problems: list[str] = []
    final: list[dict] = []
    dropped = 0
    for r in candidates:
        if r["is_persistent_constraint"] is not None:  # injected / SWE-Natural: already labeled
            final.append(r)
            continue
        row = first.get(r["id"])
        label = parse_label(row["label"]) if row else None
        if label is None:
            if args.allow_unlabeled:
                dropped += 1
                continue
            problems.append(f"{r['id']}: no label")
            continue
        r = dict(r)
        r["is_persistent_constraint"] = label
        r["labeler"] = (row.get("labeler") or "").strip() or None
        r["label_source"] = "llm" if (r["labeler"] or "").lower() == "claude" else "human"
        r["notes"] = (row.get("notes") or "").strip() or None
        if label:
            action = (row.get("constraint_action") or "add").strip().lower()
            if action not in ACTIONS:
                problems.append(f"{r['id']}: constraint_action must be add/modify/revoke, got {action!r}")
            text = (row.get("constraint_text") or "").strip()
            if not text:
                problems.append(f"{r['id']}: positive label needs constraint_text")
            r["constraint_action"] = action
            r["gold_constraint"] = text or None
            r["sc_type"] = (row.get("sc_type") or "").strip() or None
            r["sampled_as"] = r["category"]  # random_turn / keyword_flagged_turn
            r["category"] = "natural_sc_" + r["source"]
        final.append(r)

    if problems:
        print(f"{len(problems)} problem(s) in to_label.csv — fix and re-run:")
        for p in problems[:50]:
            print("  -", p)
        raise SystemExit(1)

    # Agreement between the two labelers
    agreement: dict = {}
    second_path = args.build_dir / "to_label_second.csv"
    if second_path.exists():
        second = read_sheet(second_path)
        pairs, disagreements = [], []
        for rid, row in second.items():
            b = parse_label(row["label"])
            a = parse_label(first.get(rid, {}).get("label"))
            if a is None or b is None:
                continue
            pairs.append((a, b))
            if a != b:
                disagreements.append({"id": rid, "first": int(a), "second": int(b),
                                      "user_message": row["user_message"][:2000],
                                      "first_notes": first[rid].get("notes", ""),
                                      "second_notes": row.get("notes", "")})
        if pairs:
            a, b = zip(*pairs)
            agreement = {"n": len(pairs),
                         "percent_agreement": round(100 * sum(x == y for x, y in pairs) / len(pairs), 1),
                         "cohen_kappa": round(cohen_kappa(list(a), list(b)), 3),
                         "disagreements": len(disagreements)}
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with (args.out_dir / "disagreements.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "first", "second", "user_message",
                                                  "first_notes", "second_notes"])
                w.writeheader()
                w.writerows(disagreements)
        else:
            agreement = {"n": 0, "note": "second pass not labeled yet"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        write_jsonl(args.out_dir / f"{split}.jsonl", [r for r in final if r["split"] == split])

    # Stream labels: keyword-flagged turns come from the sheet; unflagged turns are
    # assumed negative (spot-check a few before trusting this).
    stream = read_jsonl(args.build_dir / "stream_turns.jsonl")
    stream_sheet_path = args.build_dir / "stream_to_label.csv"
    stream_sheet = read_sheet(stream_sheet_path) if stream_sheet_path.exists() else {}
    unresolved = 0
    for r in stream:
        if r["keyword_hit"]:
            lab = parse_label(stream_sheet.get(r["id"], {}).get("label"))
            if lab is None:
                unresolved += 1
            r["is_persistent_constraint"] = lab
        else:
            r["is_persistent_constraint"] = False
    write_jsonl(args.out_dir / "stream_turns.jsonl", stream)
    if (args.build_dir / "stream_injections.jsonl").exists():
        shutil.copy(args.build_dir / "stream_injections.jsonl", args.out_dir / "stream_injections.jsonl")

    # Summary
    summary = {}
    for split in ("val", "test"):
        rows = [r for r in final if r["split"] == split]
        summary[split] = {
            "total": len(rows),
            "positive": sum(bool(r["is_persistent_constraint"]) for r in rows),
            "negative": sum(r["is_persistent_constraint"] is False for r in rows),
            "by_category": dict(Counter(r["category"] for r in rows)),
            "positives_by_sc_type": dict(Counter(r["sc_type"] for r in rows if r["is_persistent_constraint"])),
            "positives_by_action": dict(Counter(r["constraint_action"] for r in rows
                                                if r["is_persistent_constraint"])),
        }
    summary["stream"] = {"turns": len(stream),
                         "natural_positives": sum(r["is_persistent_constraint"] is True for r in stream),
                         "flagged_unlabeled": unresolved}
    summary["agreement"] = agreement
    summary["dropped_unlabeled"] = dropped
    (args.out_dir / "agreement.json").write_text(json.dumps(agreement, indent=2))

    build_stats = json.loads((args.build_dir / "build_stats.json").read_text())
    lines = ["# Gate dataset — datasheet", "",
             f"Source: Hugging Face `{build_stats['hf_repo']}` (CONPACT, Wang et al. 2026).", "",
             "## Source pools", ""]
    for src in ("wildchat", "hermes"):
        if src in build_stats:
            s = build_stats[src]
            lines.append(f"- **{src}**: {s['histories']} histories, {s['user_turns_after_dedupe']} unique user "
                         f"turns ({s['duplicates_removed']} duplicates removed, {s['non_english_turns']} "
                         f"non-English, {s['keyword_hit_turns']} keyword-flagged)")
    lines += ["", "## Labeled splits", ""]
    for split in ("val", "test"):
        s = summary[split]
        lines.append(f"- **{split}**: {s['total']} items, {s['positive']} positive / {s['negative']} negative; "
                     f"categories {s['by_category']}; positive SC types {s['positives_by_sc_type']}")
    lines += ["", "## Stream set (realistic prevalence)", "",
              f"- {summary['stream']['turns']} consecutive user turns from "
              f"{build_stats.get('stream', {}).get('histories', 0)} test histories; "
              f"{summary['stream']['natural_positives']} natural constraints found; "
              f"{build_stats.get('stream', {}).get('injections', 0)} injected-SC instances.",
              "", "## Label quality", ""]
    if agreement.get("n"):
        lines.append(f"- Double-labeled {agreement['n']} items: {agreement['percent_agreement']}% agreement, "
                     f"Cohen's kappa = {agreement['cohen_kappa']}.")
    else:
        lines.append("- Not double-labeled; no inter-annotator agreement measured.")
    labelers = Counter(r.get("labeler") for r in final if r.get("label_source") in ("llm", "human"))
    if labelers:
        who = ", ".join(f"{k or 'unknown'} ({v} items)" for k, v in labelers.items())
        lines.append(f"- Sampled turns labeled by: {who}, following LABELING_GUIDE.md. "
                     "Items labeled by an LLM are marked label_source = \"llm\".")
    lines += ["", "## Split rule", "",
              "Splits are by history (no conversation appears in both). Validation uses SC ids 1, 4, 7, 10, 13 "
              "(one per type); test uses the other ten, so threshold tuning never sees the test constraints' "
              "wording. Turn texts are de-duplicated within and across splits.", ""]
    if build_stats.get("warnings"):
        lines += ["## Build warnings", ""] + [f"- {w}" for w in build_stats["warnings"]] + [""]
    (args.out_dir / "DATASHEET.md").write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

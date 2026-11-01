"""Label a sheet in the terminal, one turn at a time. Saves after every answer,
so you can quit (q) and resume later; it starts at the first unlabeled row.

  python label_cli.py ../data/build/to_label.csv --labeler TM

Keys:  1 = persistent constraint   0 = not   ? = unsure (stays unlabeled)
       f = show the full message   b = back one row   q = save and quit
"""

from __future__ import annotations

import argparse
import csv
import shutil
import textwrap
from pathlib import Path

PREVIEW = 1500
TYPES = {"a": "Action", "i": "Information", "p": "Process", "r": "Preference", "o": "Output"}
ACTIONS = {"a": "add", "m": "modify", "r": "revoke", "": "add"}


def save(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def show(row: dict, i: int, n: int, full: bool) -> None:
    print("\n" + "=" * 78)
    print(f"[{i + 1}/{n}]  {row['id']}")
    prev = (row.get("previous_assistant_preview") or "").strip()
    if prev:
        print("\n--- previous assistant (end) ---")
        print(textwrap.shorten(prev, 400, placeholder=" ..."))
    msg = row["user_message"]
    print("\n--- USER MESSAGE ---")
    if not full and len(msg) > PREVIEW:
        print(msg[:PREVIEW] + f"\n... [{len(msg) - PREVIEW} more chars, press f to see all]")
    else:
        print(msg)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", type=Path)
    ap.add_argument("--labeler", required=True, help="Your initials")
    args = ap.parse_args()

    backup = args.sheet.with_name(args.sheet.stem + ".backup.csv")
    if not backup.exists():
        shutil.copy(args.sheet, backup)

    with args.sheet.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    n = len(rows)
    i = next((k for k, r in enumerate(rows) if (r.get("label") or "").strip() not in ("0", "1")), n)
    done = sum((r.get("label") or "").strip() in ("0", "1") for r in rows)
    print(f"{done}/{n} already labeled. Starting at row {i + 1}. Keys: 1 / 0 / ? / f / b / q")

    full = False
    while i < n:
        row = rows[i]
        show(row, i, n, full)
        current = (row.get("label") or "").strip()
        key = input(f"label{' (now ' + current + ')' if current else ''} > ").strip().lower()
        full = False
        if key == "q":
            break
        if key == "f":
            full = True
            continue
        if key == "b":
            i = max(0, i - 1)
            continue
        if key not in ("1", "0", "?"):
            print("Use 1, 0, ?, f, b or q.")
            continue

        row["labeler"] = args.labeler
        if key == "?":
            row["label"] = "?"
            row["notes"] = (row.get("notes") or "") + " unsure"
        elif key == "0":
            row["label"] = "0"
            row["constraint_action"] = row["constraint_text"] = row["sc_type"] = ""
        else:
            row["label"] = "1"
            text = ""
            while not text:
                text = input("  constraint, one sentence (e.g. 'Respond in Spanish.') > ").strip()
            row["constraint_text"] = text
            act = input("  action [Enter=add, m=modify, r=revoke] > ").strip().lower()
            row["constraint_action"] = ACTIONS.get(act, "add")
            t = input("  type [a=Action i=Information p=Process r=Preference o=Output, Enter=skip] > ")
            row["sc_type"] = TYPES.get(t.strip().lower(), "")
        note = input("  note (Enter to skip; e.g. autogpt, persona) > ").strip()
        if note:
            row["notes"] = ((row.get("notes") or "") + " " + note).strip()
        save(args.sheet, fields, rows)
        i += 1

    save(args.sheet, fields, rows)
    done = sum((r.get("label") or "").strip() in ("0", "1") for r in rows)
    print(f"\nSaved. {done}/{n} labeled. Backup of the original: {backup.name}")


if __name__ == "__main__":
    main()

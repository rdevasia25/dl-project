"""Copy the labeled sheets from labels/ into data/build/, after checking that their
row ids match the fresh build (they will, because the build is deterministic)."""
import csv, shutil, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
build, labels = root / "data" / "build", root / "labels"
for name in ("to_label.csv", "stream_to_label.csv"):
    fresh = {r["id"] for r in csv.DictReader(open(build / name, encoding="utf-8"))}
    labeled = {r["id"] for r in csv.DictReader(open(labels / name, encoding="utf-8"))}
    if fresh != labeled:
        sys.exit(f"{name}: ids differ from the fresh build "
                 f"({len(fresh - labeled)} new, {len(labeled - fresh)} missing). "
                 "Did the build use different settings?")
    shutil.copy(labels / name, build / name)
    print(f"{name}: {len(labeled)} labeled rows applied")
second = build / "to_label_second.csv"
if second.exists():
    second.unlink()
    print("to_label_second.csv removed (second pass skipped)")

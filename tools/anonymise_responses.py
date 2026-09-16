#!/usr/bin/env python3
"""Strip identifying data from the study responses before they go in the artifact.

The raw Google Sheet carries student registration numbers and submission
timestamps. Neither is needed to reproduce the scoring, and both identify real
people, so this rewrites participants as P01, P02, ... in submission order and
blanks the timestamps. Run it after every fresh download.

Usage:  python tools/anonymise_responses.py
In:     study/form_responses.csv            (raw, never committed)
Out:    study/form_responses_anonymised.csv (safe to publish)
"""
import csv
import os

SRC = "study/form_responses.csv"
DST = "study/form_responses_anonymised.csv"

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
if not rows:
    raise SystemExit(f"{SRC} is empty.")
fields = list(rows[0])
pid = next(c for c in fields if c.strip().lower().startswith(("registration", "participant")))
stamp = next((c for c in fields if "timestamp" in c.lower()), None)

for i, r in enumerate(rows, 1):
    r[pid] = f"P{i:02d}"
    if stamp:
        r[stamp] = ""

with open(DST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

leaked = [r[pid] for r in rows if not r[pid].startswith("P")]
assert not leaked, leaked
print(f"wrote {DST}: {len(rows)} participants, registration numbers and timestamps removed")

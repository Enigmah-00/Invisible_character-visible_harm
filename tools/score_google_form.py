#!/usr/bin/env python3
"""Score the perceptibility study from the Google Sheet.

In the response Sheet: File > Download > Comma-separated values (.csv).
Save it as study/form_responses.csv, then run this script.

A participant counts only if they catch at least 4 of the 5 control pairs that
genuinely look different. Anyone answering "same" to everything fails that check,
so they cannot inflate the result in our favour.
"""
import csv
import math
import re
import sys

KEY = {int(r["trial"]): r for r in csv.DictReader(open("study/form_answer_key.csv"))}
try:
    rows = list(csv.DictReader(open("study/form_responses.csv", encoding="utf-8-sig")))
except FileNotFoundError:
    sys.exit("study/form_responses.csv not found. Download the Sheet as CSV and save it there.")
if not rows:
    sys.exit("The response sheet is empty.")

cols = {}
for c in rows[0]:
    m = re.match(r"\s*Trial\s+(\d+)", c)
    if m and "look the same" in c:
        cols[int(m.group(1))] = c
if len(cols) != len(KEY):
    sys.exit(f"Expected {len(KEY)} trial columns, found {len(cols)}. Was the form edited?")

pid_col = next(c for c in rows[0]
               if c.strip().lower().startswith(("registration", "participant")))
daily_col = next((c for c in rows[0] if "every day" in c.lower()), None)
only_daily = "--daily" in sys.argv


def answer(v):
    v = (v or "").upper()
    return "DIFFERENT" if "DIFFERENT" in v else "SAME" if "SAME" in v else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


by_kind = {k: [t for t, v in KEY.items() if v["kind"] == k] for k in ("test", "control", "control-same")}
kept, dropped, skipped = [], [], []
for r in rows:
    pid = r[pid_col].strip() or "(blank)"
    if only_daily and daily_col and r[daily_col].strip().lower() != "yes":
        skipped.append(pid)
        continue
    ans = {t: answer(r[c]) for t, c in cols.items()}
    caught = sum(ans[t] == "DIFFERENT" for t in by_kind["control"])
    (kept if caught >= 4 else dropped).append((pid, caught, ans))

print(f"responses: {len(rows)}   kept: {len(kept)}   "
      f"excluded for failing controls: {len(dropped)}"
      + (f"   skipped (not daily readers): {len(skipped)}" if only_daily else ""))
for pid, caught, _ in dropped:
    print(f"    excluded {pid}: caught only {caught}/{len(by_kind['control'])} real differences")
if not kept:
    sys.exit("\nNo participant passed the control check. Nothing to report.")


def tally(kind):
    hit = tot = 0
    for _, _, ans in kept:
        for t in by_kind[kind]:
            if ans[t] is not None:
                tot += 1
                hit += ans[t] == "DIFFERENT"
    return hit, tot


det, tot = tally("test")
cd, ct = tally("control")
fa, ft = tally("control-same")
lo, hi = wilson(det, tot)

print()
print(f"ATTACKED pairs judged different : {det}/{tot} = {det/tot:.1%}   [95% CI {lo:.1%} to {hi:.1%}]")
print(f"REAL differences caught         : {cd}/{ct} = {cd/ct:.1%}")
print(f"IDENTICAL images called different: {fa}/{ft} = {fa/ft:.1%}   (false-alarm baseline)")
print()
if det / tot <= fa / ft + 0.05:
    verdict = "attacked pairs were no more often called different than identical images"
else:
    verdict = "attacked pairs were called different more often than identical images - look closer"
print("Verdict:", verdict)
print()
print("Sentence for the paper:")
print(f'  "{len(kept)} Bangla readers who passed a control check judged {det} of {tot} attacked')
print(f'   pairs ({det/tot:.1%}, 95% CI {lo:.1%}-{hi:.1%}) to look different, against {fa/ft:.1%}')
print(f'   for pairs of identical images, while catching {cd/ct:.0%} of pairs that genuinely differ."')

# ---- emit the machine-readable result the paper generator consumes -----------
# `kept` holds (pid, controls_caught, answers) tuples, so the daily-reader flag
# comes back from the original rows by registration number.
import os, csv as _csv

is_daily = {}
for _r in rows:
    _pid = (_r[pid_col] or "").strip()
    is_daily[_pid] = bool(daily_col) and _r[daily_col].strip().lower().startswith("y")


def _sub(entries, kind):
    hit = tot = 0
    for _pid, _caught, ans in entries:
        for t in by_kind[kind]:
            tot += 1
            hit += ans[t] == "DIFFERENT"
    return hit, tot


daily = [e for e in kept if is_daily.get(e[0].strip(), False)]
rows_out = []
for label, entries in [("all", kept), ("daily_bangla_readers", daily)]:
    for kind, human in [("test", "attacked pair"), ("control-same", "identical images"),
                        ("control", "genuinely different pair")]:
        h, n = _sub(entries, kind)
        shown = h if kind != "control" else n - h
        lo2, hi2 = wilson(shown, n)
        rows_out.append({"group": label, "n_participants": len(entries), "stimulus": human,
                         "measure": "called different" if kind != "control" else "missed",
                         "k": shown, "n": n,
                         "rate": round(shown / n, 4) if n else None,
                         "ci_low": round(lo2, 4), "ci_high": round(hi2, 4)})

os.makedirs("results", exist_ok=True)
with open("results/human_study.csv", "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
print("\nwrote results/human_study.csv")

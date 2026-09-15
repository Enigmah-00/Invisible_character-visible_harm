#!/usr/bin/env python3
"""Score the perceptibility study.

Fill study/responses.csv with one row per participant per trial:
    participant,trial,answer        (answer is SAME or DIFFERENT)

A participant counts only if they catch at least 4 of the 5 control pairs that
genuinely differ. Someone who answers SAME to everything fails that check, so
they cannot inflate the result in our favour.
"""
import csv
import math
import sys
from collections import defaultdict

key = {int(r["trial"]): r for r in csv.DictReader(open("study/answer_key.csv"))}
try:
    rows = list(csv.DictReader(open("study/responses.csv")))
except FileNotFoundError:
    sys.exit("study/responses.csv not found - copy responses_template.csv and fill it in")

resp = defaultdict(dict)
for r in rows:
    resp[r["participant"].strip()][int(r["trial"])] = r["answer"].strip().upper()


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


ctrl_trials = [t for t, v in key.items() if v["kind"] == "control"]
test_trials = [t for t, v in key.items() if v["kind"] == "test"]
same_trials = [t for t, v in key.items() if v["kind"] == "control-same"]

kept, dropped = [], []
for pid, ans in resp.items():
    hit = sum(1 for t in ctrl_trials if ans.get(t) == "DIFFERENT")
    (kept if hit >= 4 else dropped).append((pid, hit))

print(f"participants: {len(resp)}   kept: {len(kept)}   excluded for failing controls: {len(dropped)}")
for pid, hit in dropped:
    print(f"    excluded {pid}: caught only {hit}/{len(ctrl_trials)} controls")
if not kept:
    sys.exit("\nno participant passed the control check - nothing to report")

det = tot = 0
for pid, _ in kept:
    ans = resp[pid]
    for t in test_trials:
        if t in ans:
            tot += 1
            det += ans[t] == "DIFFERENT"
lo, hi = wilson(det, tot)

cd = ct = 0
for pid, _ in kept:
    for t in ctrl_trials:
        if t in resp[pid]:
            ct += 1
            cd += resp[pid][t] == "DIFFERENT"

fp = fpt = 0
for pid, _ in kept:
    for t in same_trials:
        if t in resp[pid]:
            fpt += 1
            fp += resp[pid][t] == "DIFFERENT"

print()
print(f"ATTACKED pairs detected as different : {det}/{tot} = {det/tot:.1%}  [95% CI {lo:.1%}-{hi:.1%}]")
print(f"CONTROL pairs detected as different  : {cd}/{ct} = {cd/ct:.1%}   (people can see real differences)")
if fpt:
    print(f"IDENTICAL pairs called different     : {fp}/{fpt} = {fp/fpt:.1%}   (baseline false-alarm rate)")
print()
print("Sentence for the paper:")
print(f'  "Across {len(kept)} readers of Bangla who passed a control check, {det} of {tot}')
print(f'   attacked pairs ({det/tot:.1%}, 95% CI {lo:.1%}-{hi:.1%}) were judged to look different,')
print(f'   while {cd/ct:.0%} of control pairs that genuinely differ were correctly identified."')

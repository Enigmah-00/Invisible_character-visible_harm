"""Overlap of BLP-2025 Subtask 1A with BD-SHS and the 30k set. Counts only, never comment text."""
import sys, os, re, glob, unicodedata, csv
import pandas as pd
root = sys.argv[1]
def exact(t): return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()
def loose(t): return "".join(ch for ch in unicodedata.normalize("NFC", str(t)) if ch.isalnum())
blp = {}
for split in ("train", "dev", "test_with_labels"):
    p = os.path.join(root, "blp25", f"blp25_hatespeech_subtask_1A_{split}.tsv")
    d = pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    blp[split] = d
    print(f"BLP25 {split:16s}: {len(d):6d} rows | columns {list(d.columns)} | labels {d['label'].str.strip('\"').value_counts().to_dict() if 'label' in d else '-'}")
bd = pd.concat([pd.read_csv(os.path.join(root, "bdshs", f"{s}.csv")) for s in ("train", "val", "test")])
r30 = pd.read_csv(glob.glob(os.path.join(root, "bhsd30k", "*.csv"))[0])
for name, f in (("exact", exact), ("loose", loose)):
    bk = set(k for k in bd["sentence"].map(f) if len(k) >= 10)
    rk = set(k for k in r30["sentence"].map(f) if len(k) >= 10)
    for s, d in blp.items():
        k = set(x for x in d["text"].map(f) if len(x) >= 10)
        print(f"{name:5s} {s:16s}: unique {len(k):6d} | in BD-SHS {len(k & bk):5d} ({len(k & bk)/len(k):.2%}) | in 30k {len(k & rk):5d} ({len(k & rk)/len(k):.2%})")
    tr = set(x for x in blp["train"]["text"].map(f) if len(x) >= 10)
    te = set(x for x in blp["test_with_labels"]["text"].map(f) if len(x) >= 10)
    print(f"{name:5s} BLP train/test leakage: {len(tr & te)} shared ({len(tr & te)/len(te):.2%} of test)")

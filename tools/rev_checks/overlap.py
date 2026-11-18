"""Overlap between BD-SHS and the Romim et al. 30k set. Prints counts only, never comment text."""
import sys, glob, os, re, unicodedata
import pandas as pd
root = sys.argv[1]

def load(path):
    d = pd.read_csv(path)
    obj = [c for c in d.columns if d[c].dtype == object or pd.api.types.is_string_dtype(d[c])]
    text = max(obj, key=lambda c: d[c].astype(str).str.len().mean())
    label = [c for c in d.columns if c != text and d[c].dropna().nunique() == 2][0]
    return d, text, label

def exact(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()

def loose(t):  # letters and digits only: ignores punctuation, emoji, spacing, invisible characters
    return "".join(ch for ch in unicodedata.normalize("NFC", str(t)) if ch.isalnum())

bd = {}
for split in ("train", "val", "test"):
    d, t, l = load(os.path.join(root, "bdshs", f"{split}.csv"))
    bd[split] = d
    print(f"BD-SHS {split:5s}: {len(d):6d} rows | text col '{t}' | label col '{l}' | columns {list(d.columns)}")
p30 = glob.glob(os.path.join(root, "bhsd30k", "*.csv"))[0]
r, rt, rl = load(p30)
print(f"30k set      : {len(r):6d} rows | text col '{rt}' | label col '{rl}' | columns {list(r.columns)}")

bd_all = pd.concat([d.assign(_split=s) for s, d in bd.items()], ignore_index=True)
_, bt, bl = load(os.path.join(root, "bdshs", "train.csv"))
for name, f in (("exact (spacing/NFC normalized)", exact), ("loose (letters+digits only)", loose)):
    bkeys = bd_all[bt].map(f)
    rkeys = r[rt].map(f)
    rk = set(k for k in rkeys if len(k) >= 10)       # ignore very short strings like "ok"
    bk = set(k for k in bkeys if len(k) >= 10)
    inter = rk & bk
    print(f"\n{name}:")
    print(f"  unique 30k texts {len(rk)} | unique BD-SHS texts {len(bk)} | shared {len(inter)} "
          f"= {len(inter)/len(rk):.1%} of 30k, {len(inter)/len(bk):.1%} of BD-SHS")
    for s in ("train", "val", "test"):
        ks = set(k for k in bd[s][bt].map(f) if len(k) >= 10)
        print(f"  shared with BD-SHS {s:5s}: {len(rk & ks)} ({len(rk & ks)/max(1,len(ks)):.1%} of that split)")
    # do shared comments carry the same hate label?
    rl_map = dict(zip(rkeys, r[rl])); bl_map = dict(zip(bkeys, bd_all[bl]))
    same = sum(int(rl_map[k]) == int(bl_map[k]) for k in inter)
    print(f"  label agreement on shared: {same}/{len(inter)} ({same/max(1,len(inter)):.1%})")
print("\n30k internal duplicates (exact):", len(r) - r[rt].map(exact).nunique())

"""Experiments for the ICCIT 2026 revision.

One job is (dataset, model, seed). The nb14_* notebooks call prepare() once per
dataset to build the attack cache on CPU, then start one worker per T4 GPU:

    python revision_run.py --lane '[["bdshs", "xlmr", 42, "full"], ...]'

Each job writes <dataset>_<model>_s<seed>_<table>.csv into OUT. The training
protocol is the paper's Table III one, unchanged: full training split, effective
batch 32, 2 epochs, max length 128, fp32, transformers 4.46.3.

New relative to the submitted paper:
  - attacks with other invisible characters (CGJ, VS16, LRM), which the
    WordPiece cleaning step does not remove (CGJ, VS16) or the old defense
    does not know about (all three);
  - the defenses generalized to every default-ignorable code point;
  - a decision-change rule for the residual flag (flag only when the leftover
    characters change the model's answer);
  - extra seeds and a second dataset.
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_attack as ra  # noqa: E402

OUT = os.environ.get("REV_OUT", "/kaggle/working/out")
INPUT = os.environ.get("REV_INPUT", "/kaggle/input")
SMOKE = os.environ.get("REV_SMOKE") == "1"   # tiny run for a local code check

HATE, MAXLEN, EFF_BATCH, EPOCHS = 1, 128, 32, 2
N_ATK = 1500
MODELS = {  # short name: (hub id, tokenizer family, learning rate), as in Table III
    "banglabert": ("csebuetnlp/banglabert", "WordPiece", 2e-5),
    "muril": ("google/muril-base-cased", "WordPiece", 2e-5),
    "xlmr": ("xlm-roberta-base", "SentencePiece", 2e-5),
    "mdeberta": ("microsoft/mdeberta-v3-base", "SentencePiece", 1e-5),
    "banglat5": ("csebuetnlp/banglat5", "SentencePiece", 1e-4),
}
# Kaggle input folder of each Kaggle-hosted dataset. (The authors' earlier 30k set,
# naurosromim/bengali-hate-speech-dataset, is NOT a second dataset: 69% of its
# comments are also in BD-SHS.)
DATASETS = {"bdshs": "bdshs"}
# BLP-2025 shared task, subtask 1A: Bangla YouTube comments labelled Abusive /
# Sexism / Religious Hate / Political Hate / Profane / None, here hate = not None.
# Independent of BD-SHS: 10 of its 45,691 comments occur there. Hosted on GitHub.
BLP25_URL = ("https://raw.githubusercontent.com/AridHasan/blp25_task1/main/data/subtask_1A/"
             "blp25_hatespeech_subtask_1A_{}.tsv")
ATTACK_CHARS = {"zwnj": ra.ZWNJ, "cgj": ra.CGJ, "vs16": ra.VS16, "lrm": ra.LRM}
DI = ra.is_default_ignorable


# ------------------------------------------------------------------ data

def load_blp25():
    import csv
    import urllib.request
    folder = os.path.join(OUT, "data_blp25")
    os.makedirs(folder, exist_ok=True)
    out = []
    for split in ("train", "test_with_labels"):
        p = os.path.join(folder, f"{split}.tsv")
        if not os.path.exists(p):
            urllib.request.urlretrieve(BLP25_URL.format(split), p)
        # keep_default_na=False: otherwise pandas reads the label "None" as missing
        d = pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
        d = pd.DataFrame({"text": d["text"].astype(str),
                          "label": (d["label"].str.strip().str.strip('"') != "None").astype(int)})
        out.append(d[d.text.str.strip().str.len() > 0].reset_index(drop=True))
    return out


def load_dataset(ds):
    if ds == "blp25":
        train, test = load_blp25()
        if SMOKE:
            train = train.sample(64, random_state=0).reset_index(drop=True)
            test = test.sample(64, random_state=0).reset_index(drop=True)
        return train, test
    # Kaggle mounts a dataset somewhere under /kaggle/input, in a folder named after its slug
    files = sorted(f for f in glob.glob(os.path.join(INPUT, "**", "*.csv"), recursive=True)
                   if f"/{DATASETS[ds]}/" in f)
    if not files:
        raise FileNotFoundError(f"no CSV for {DATASETS[ds]} under {INPUT} - add the dataset as an input")
    named = {k: [f for f in files if k in os.path.basename(f).lower()] for k in ("train", "test")}

    def prep(d, text=None, label=None):
        obj = [c for c in d.columns if d[c].dtype == object or pd.api.types.is_string_dtype(d[c])]
        text = text or max(obj, key=lambda c: d[c].astype(str).str.len().mean())
        label = label or [c for c in d.columns if c != text and d[c].dropna().nunique() == 2][0]
        d = d[[text, label]].dropna().copy()
        d.columns = ["text", "label"]
        d["text"] = d["text"].astype(str)
        d["label"] = d["label"].astype(int)
        return d[d.text.str.strip().str.len() > 0].reset_index(drop=True), text, label

    if named["train"] and named["test"]:
        train, t, l = prep(pd.read_csv(named["train"][0]))
        test, _, _ = prep(pd.read_csv(named["test"][0]), t, l)
    else:
        # one file: a fixed stratified 80/20 split
        from sklearn.model_selection import train_test_split
        full, _, _ = prep(pd.read_csv(max(files, key=os.path.getsize)))
        full = full.drop_duplicates("text").reset_index(drop=True)
        train, test = train_test_split(full, test_size=0.2, random_state=42, stratify=full.label)
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    if SMOKE:
        train = train.sample(min(64, len(train)), random_state=0).reset_index(drop=True)
        test = test.sample(min(64, len(test)), random_state=0).reset_index(drop=True)
    return train, test


def cache_path(ds):
    return os.path.join(OUT, f"cache_{ds}.json")


def tag_of(ds, short, seed, level):
    return f"{ds}_{short}{'_adv' if level == 'advtrain' else ''}_s{seed}"


# ------------------------------------------------------------------ CPU stage

def defenses(R):
    return {
        "none": lambda t: t,
        "nfc": ra.defend_nfc,
        "strip_all": ra.defend_strip_all,
        "position_aware": ra.defend_position_aware,
        "render_equivalence": R.defend_render_equivalence,
        "strip_all_di": lambda t: ra.defend_strip_all(t, DI),
        "render_equivalence_di": lambda t: R.defend_render_equivalence(t, invisible=DI),
    }


def prepare(ds, R):
    """Attacks, defended texts and wild comments for one dataset, cached to JSON."""
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(cache_path(ds)):
        return json.load(open(cache_path(ds), encoding="utf-8"))
    t0 = time.time()
    train, test = load_dataset(ds)
    n = min(N_ATK if not SMOKE else 16, (test.label == HATE).sum(), (test.label != HATE).sum())
    rng = random.Random(0)   # same draw as every earlier step
    hate = [test.text[i] for i in sorted(rng.sample(test.index[test.label == HATE].tolist(), n))]
    ben = [test.text[i] for i in sorted(rng.sample(test.index[test.label != HATE].tolist(), n))]

    atk, meta = {}, []
    for k, c in ATTACK_CHARS.items():
        res = [R.attack(t, "ALL", char=c, mode="inert", seed=i) for i, t in enumerate(hate)]
        atk[k] = [a for a, _ in res]
        meta.append({"attack": k, "mean_inserted": round(float(np.mean([m for _, m in res])), 2)})
    # the visible attack, for the review-burden analysis
    atk["zwnj_ungated"] = [ra.attack_ungated(t, "ALL", mode="inert", seed=i)[0] for i, t in enumerate(hate)]
    meta.append({"attack": "zwnj_ungated", "mean_inserted": None})
    for m in meta:
        m["visible_any_font"] = round(float(np.mean(
            [len(R.visible_in(h, a)) > 0 for h, a in zip(hate, atk[m["attack"]])])), 4)
    ben_atk = {"zwnj": [R.attack(t, "ALL", mode="both", seed=i)[0] for i, t in enumerate(ben)],
               "cgj": [R.attack(t, "ALL", char=ra.CGJ, mode="inert", seed=i)[0] for i, t in enumerate(ben)]}

    D = defenses(R)
    defended = {f"{a}|{d}": [fn(t) for t in atk[a]] for a in atk for d, fn in D.items()}
    clean_def = {d: [fn(t) for t in test.text] for d, fn in D.items()}

    # Real comments that already contain invisible characters, from the whole corpus
    alltxt = train.text.tolist() + test.text.tolist()
    wild_zw = [t for t in alltxt if any(z in t for z in ra.ZW)]
    wild_di = [t for t in alltxt if any(DI(c) for c in t)]
    collateral = []
    for name, texts in (("zero-width", wild_zw), ("any default-ignorable", wild_di)):
        for d in ("strip_all", "position_aware", "render_equivalence", "strip_all_di", "render_equivalence_di"):
            changed = [len(R.visible_in(t, D[d](t))) > 0 for t in texts]
            collateral.append({"wild_set": name, "n": len(texts), "defense": d,
                               "rendering_changed": int(sum(changed))})

    cache = {"dataset": ds, "n_train": len(train), "n_test": len(test), "hate": hate, "ben": ben,
             "atk": atk, "ben_atk": ben_atk, "defended": defended, "clean_def": clean_def,
             "test_label": test.label.tolist(), "wild_zw": wild_zw, "wild_di": wild_di,
             "attack_meta": meta, "collateral": collateral, "seconds": round(time.time() - t0)}
    json.dump(cache, open(cache_path(ds), "w", encoding="utf-8"), ensure_ascii=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, f"{ds}_attack_meta.csv"), index=False)
    pd.DataFrame(collateral).to_csv(os.path.join(OUT, f"{ds}_collateral.csv"), index=False)
    return cache


# ------------------------------------------------------------------ GPU stage

def run_job(ds, short, seed, level):
    import torch
    from functools import lru_cache
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, f1_score
    from normalizer import normalize

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, family, lr = MODELS[short]
    tag = tag_of(ds, short, seed, level)
    C = json.load(open(cache_path(ds), encoding="utf-8"))
    train, _ = load_dataset(ds)
    if level == "advtrain":
        # Adversarial training as in Step 9: half the training comments appear a second
        # time carrying the verified ZWNJ attack. Evaluated on every attack, including the
        # characters it never saw in training.
        R = _renderer()
        idx = random.Random(0).sample(range(len(train)), len(train) // 2)
        aug = pd.DataFrame({"text": [R.attack(train.text[i], "ALL", mode="inert", seed=9000 + i)[0] for i in idx],
                            "label": [int(train.label[i]) for i in idx]})
        train = pd.concat([train, aug], ignore_index=True)
        print(f"[{tag}] adversarial training set: {len(train)} comments", flush=True)
    ty = np.array(C["test_label"])

    @lru_cache(maxsize=None)
    def norm(t):
        return normalize(t)

    class DS(Dataset):
        def __init__(self, tk, texts, labels):
            self.ids = tk([norm(t) for t in texts], truncation=True, max_length=MAXLEN)["input_ids"]
            self.y = list(labels)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.ids[i], self.y[i]

    def collate(tk):
        def fn(b):
            enc = tk.pad({"input_ids": [x[0] for x in b]}, padding=True, return_tensors="pt")
            return enc["input_ids"], enc["attention_mask"], torch.tensor([x[1] for x in b])
        return fn

    # Table III training function, verbatim apart from the seed argument
    micro, epochs = 8, (1 if SMOKE else EPOCHS)
    accum = max(1, EFF_BATCH // micro)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)
    tk = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tk.pad_token_id
    dl = DataLoader(DS(tk, train.text, train.label), batch_size=micro, shuffle=True,
                    collate_fn=collate(tk), generator=torch.Generator().manual_seed(seed))
    decay, no_decay = [], []
    for n_, p in mdl.named_parameters():
        f = n_.lower()
        (no_decay if (f.endswith(".bias") or "layernorm" in f or "layer_norm" in f) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, eps=1e-6)
    steps = (len(dl) // accum) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.1)
    mdl.train(); skipped = done = 0
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        for st, (i_, m_, y_) in enumerate(dl):
            loss = mdl(input_ids=i_.to(dev), attention_mask=m_.to(dev), labels=y_.to(dev)).loss / accum
            if torch.isfinite(loss):
                loss.backward()
            else:
                skipped += 1
            if (st + 1) % accum == 0:
                ok = all(torch.isfinite(p.grad).all() for p in mdl.parameters() if p.grad is not None)
                if ok:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
                else:
                    skipped += 1
                opt.zero_grad()
                if done < steps:
                    sch.step(); done += 1
            if st % 1500 == 0:
                print(f"[{tag}] ep{ep} {st}/{len(dl)} loss {loss.item() * accum:.4f} skipped {skipped}", flush=True)
    mdl.eval()
    train_min = round((time.time() - t0) / 60, 1)

    @torch.no_grad()
    def predict(texts, bs=16):
        if len(texts) == 0:   # e.g. a model that flags no hate leaves no targets
            return np.empty(0, dtype=int)
        ids = tk([norm(t) for t in texts], truncation=True, max_length=512)["input_ids"]
        order = np.argsort([len(z) for z in ids]); out = np.empty(len(texts), dtype=int); s = 0
        while s < len(order):
            idx = order[s:s + bs]
            try:
                b = tk.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
                lg = mdl(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).logits
                out[idx] = lg.float().argmax(-1).cpu().numpy(); s += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
        return out

    def save(rows, table):
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}_{table}.csv"), index=False)

    base = {"dataset": ds, "model": short + ("_adv" if level == "advtrain" else ""), "family": family, "seed": seed}
    full = level in ("full", "advtrain")
    D_ALL = list(defenses_names()) if full else ["none", "render_equivalence", "render_equivalence_di"]
    attacks = list(C["atk"]) if full else ["zwnj", "cgj"]

    # clean accuracy, with each defense applied to the clean test set
    clean = []
    for d in (D_ALL if full else ["none", "render_equivalence_di"]):
        p = predict(C["clean_def"][d])
        clean.append({**base, "defense_on_clean": d, "acc": round(accuracy_score(ty, p), 4),
                      "f1": round(f1_score(ty, p), 4)})
    save(clean, "clean")

    # evasion: of the hate comments the model catches, how many escape
    tgt = np.where(predict(C["hate"]) == HATE)[0]
    ev = []
    for a in attacks:
        for d in D_ALL:
            texts = C["atk"][a] if d == "none" else C["defended"][f"{a}|{d}"]
            k = int((predict([texts[i] for i in tgt]) != HATE).sum())
            ev.append({**base, "attack": a, "defense": d, "n_targets": len(tgt), "evaded": k,
                       "evasion": round(k / max(1, len(tgt)), 4)})
    save(ev, "evasion")

    # benign control: does the attack also remove flags from ordinary comments?
    bp = predict(C["ben"])
    ben = []
    for a, texts in C["ben_atk"].items():
        bpa = predict(texts)
        ben.append({**base, "attack": a, "benign_fpr_clean": round(float((bp == HATE).mean()), 4),
                    "benign_fpr_attacked": round(float((bpa == HATE).mean()), 4),
                    "benign_flip_rate": round(float((bp != bpa).mean()), 4)})
    save(ben, "benign")

    if full:
        save(review_burden(C, tgt, predict, base), "flag")
    save([{**base, "level": level, "n_train": len(train), "skipped_steps": skipped,
           "train_minutes": train_min, "transformers": __import__("transformers").__version__,
           "device": dev}], "status")
    print(f"[{tag}] done in {train_min} min of training", flush=True)


def defenses_names():
    return ("none", "nfc", "strip_all", "position_aware", "render_equivalence",
            "strip_all_di", "render_equivalence_di")


def review_burden(C, tgt, predict, base):
    """Residual flag, two rules, after the generalized defense.

    rule      - any invisible character left beside a Bangla letter (the paper's
                rule, extended to every default-ignorable character)
    decision  - the rule fires AND deleting those leftovers changes the model's
                answer; a leftover that changes nothing is not worth a reviewer
    The old pipeline (zero-width defense + zero-width flag) is reported too, to
    show what the new invisible characters do to it."""
    R = _renderer()
    rows = []
    sets = {"wild zero-width (benign)": C["wild_zw"], "wild any default-ignorable (benign)": C["wild_di"],
            "clean test (benign)": C["clean_def"]["none"]}
    for a in C["atk"]:
        sets[f"attack {a} (hate targets)"] = [C["atk"][a][i] for i in tgt]
    for name, texts in sets.items():
        if not texts:
            continue
        after = [R.defend_render_equivalence(t, invisible=DI) for t in texts]
        flag = np.array([ra.residual_flag(t, DI) for t in after])
        p_after = predict(after)
        p_strip = predict([ra.defend_strip_all(t, DI) for t in after])
        decision = flag & (p_after != p_strip)
        old_after = [R.defend_render_equivalence(t) for t in texts]
        old_flag = np.array([ra.residual_flag(t) for t in old_after])
        row = {**base, "set": name, "n": len(texts),
               "flag_rule": round(float(flag.mean()), 4), "flag_decision": round(float(decision.mean()), 4),
               "old_flag": round(float(old_flag.mean()), 4)}
        if name.startswith("attack"):
            evaded = p_after != HATE
            old_evaded = predict(old_after) != HATE
            row.update({"evaded_after_defense": round(float(evaded.mean()), 4),
                        "unmitigated_rule": round(float((evaded & ~flag).mean()), 4),
                        "unmitigated_decision": round(float((evaded & ~decision).mean()), 4),
                        "old_pipeline_unmitigated": round(float((old_evaded & ~old_flag).mean()), 4)})
        rows.append(row)
    return rows


_R = None


def _renderer():
    global _R
    if _R is None:
        _R = ra.Renderer(ra.fetch_fonts(os.path.join(OUT, "fonts")))
    return _R


if __name__ == "__main__":
    lane = json.loads(sys.argv[sys.argv.index("--lane") + 1])
    for ds, short, seed, level in lane:
        tag = tag_of(ds, short, seed, level)
        if os.path.exists(os.path.join(OUT, f"{tag}_status.csv")):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        try:
            run_job(ds, short, int(seed), level)
        except Exception as ex:  # one failed model must not stop the lane
            import traceback
            traceback.print_exc()
            pd.DataFrame([{"dataset": ds, "model": short, "seed": seed, "level": level,
                           "error": f"{type(ex).__name__}: {str(ex)[:300]}"}]).to_csv(
                os.path.join(OUT, f"{tag}_error.csv"), index=False)
        finally:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

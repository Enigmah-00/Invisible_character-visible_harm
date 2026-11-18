#!/usr/bin/env python3
"""Write the Kaggle notebooks for the ICCIT 2026 revision.

    python3 tools/make_revision_notebooks.py

Each notebook embeds the current src/render_attack.py and src/revision_run.py,
so rerun this after editing either file. Outputs go to notebooks/nb14*.ipynb
and notebooks/nb15_revision_llms.ipynb.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {n: open(os.path.join(ROOT, "src", n), encoding="utf-8").read()
       for n in ("render_attack.py", "revision_run.py")}

INSTALL = '''# Pin the library BEFORE anything imports it - same pin as Table III (Step 12/13).
!pip install -q "transformers==4.46.3" "tokenizers<0.21" "emoji==1.7.0" regex ftfy uharfbuzz sentencepiece protobuf
!pip install -q git+https://github.com/csebuetnlp/normalizer
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__, "| GPUs", torch.cuda.device_count())
assert transformers.__version__ == "4.46.3", "wrong version - restart the session and run again"'''

PREPARE = '''import os, sys, json, glob, time
sys.path.insert(0, "/kaggle/working")
import render_attack as ra, revision_run as rr
os.makedirs(rr.OUT, exist_ok=True)
R = rr._renderer()
# self-test: every new attack is invisible and the generalized defense undoes it exactly
s = "আমাদের দেশ বাংলাদেশ ধন্যবাদ স্যার"
for k, c in rr.ATTACK_CHARS.items():
    a, n = R.attack(s, "ALL", char=c, mode="inert")
    assert n > 0 and R.same(s, a) and R.defend_render_equivalence(a, invisible=ra.is_default_ignorable) == s, k
print("self-test ok | fonts:", len(R.names))
for ds in DATASETS_USED:
    t0 = time.time(); C = rr.prepare(ds, R)
    print(f"{ds}: train {C['n_train']} | test {C['n_test']} | attacked {len(C['hate'])} | "
          f"wild zero-width {len(C['wild_zw'])} | wild any invisible {len(C['wild_di'])} | {time.time() - t0:.0f}s")'''

LAUNCH = '''import subprocess, json, time, os, sys
COST = {"banglabert": 1.0, "muril": 1.3, "xlmr": 1.3, "mdeberta": 2.2, "banglat5": 2.6}
lanes, load = [[], []], [0.0, 0.0]
for job in sorted(JOBS, key=lambda j: -COST[j[1]]):
    g = load.index(min(load)); lanes[g].append(job); load[g] += COST[job[1]]
procs = []
for g, lane in enumerate(lanes):
    if not lane: continue
    log = open(f"{rr.OUT}/lane{g}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TOKENIZERS_PARALLELISM": "false"}
    procs.append(subprocess.Popen([sys.executable, "-u", "/kaggle/working/revision_run.py", "--lane", json.dumps(lane)],
                                  env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f"GPU {g}: {[j[1] + '/s' + str(j[2]) for j in lane]}")
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(300)
    for g in range(len(procs)):
        tail = [l for l in open(f"{rr.OUT}/lane{g}.log").read().splitlines() if l.startswith("[")][-1:]
        print(f"{(time.time() - t0) / 60:6.0f} min | GPU {g} | {tail[0] if tail else '...'}")
print("all lanes finished in", round((time.time() - t0) / 60), "min")'''

SUMMARY = '''import pandas as pd, glob, shutil
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
def cat(table):
    fs = sorted(glob.glob(f"{rr.OUT}/*_s*_{table}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
print("=" * 100); print(TITLE); print("=" * 100)
err = cat("error")
if len(err): print("\\n-- FAILED JOBS --"); print(err.to_string(index=False))
st = cat("status"); print("\\n-- jobs --"); print(st.to_string(index=False))
cl = cat("clean"); print("\\n-- clean accuracy (no defense) --"); print(cl[cl.defense_on_clean == "none"].to_string(index=False))
ev = cat("evasion")
if len(ev):
    print("\\n-- evasion, no defense --")
    print(ev[ev.defense == "none"].pivot_table(index=["dataset", "model", "seed"], columns="attack", values="evasion").to_string())
    print("\\n-- evasion after the generalized defense --")
    print(ev[ev.defense == "render_equivalence_di"].pivot_table(index=["dataset", "model", "seed"], columns="attack", values="evasion").to_string())
fl = cat("flag")
if len(fl): print("\\n-- review burden --"); print(fl.to_string(index=False))
for t in ("clean", "evasion", "benign", "flag", "status"):
    d = cat(t)
    if len(d): d.to_csv(f"/kaggle/working/{ZIPNAME}_{t}.csv", index=False)
shutil.make_archive(f"/kaggle/working/{ZIPNAME}", "zip", rr.OUT)
print(f"\\nDownload /kaggle/working/{ZIPNAME}.zip from the Output tab and send it to me.")
print("=" * 100)'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}}}


def training_nb(title, intro, datasets, jobs, zipname):
    cfg = (f"DATASETS_USED = {datasets!r}\n"
           f"JOBS = {json.dumps(jobs)}  # (dataset, model, seed, level)\n"
           f"TITLE = {title!r}\nZIPNAME = {zipname!r}")
    return notebook([
        md(intro), code(INSTALL),
        code("%%writefile /kaggle/working/render_attack.py\n" + SRC["render_attack.py"]),
        code("%%writefile /kaggle/working/revision_run.py\n" + SRC["revision_run.py"]),
        code(cfg), code(PREPARE), code(LAUNCH), code(SUMMARY)])


SETTINGS = ("**Settings:** Accelerator **GPU T4 x2**, Internet **ON**.  \n"
            "Then **Save Version -> Save & Run All (Commit)** and close the tab; it runs in the background.  \n"
            "When it finishes, open the version, go to **Output**, download the `.zip` and send it to me.")

ALL5 = ["banglabert", "muril", "xlmr", "mdeberta", "banglat5"]
NBS = {
    "nb14a_revision_bdshs_seed42.ipynb": training_nb(
        "REVISION 14a - BD-SHS, seed 42, all five models, full evaluation",
        "# Revision 14a: new attacks, generalized defense, review burden (BD-SHS, seed 42)\n\n"
        "Retrains the five Table III models (the weights were never saved) and evaluates the reviewer requests:\n"
        "attacks with CGJ / VS16 / LRM, the defense generalized to every invisible character, and the\n"
        "decision-change rule for the residual flag.\n\n"
        "**Inputs:** dataset `naurosromim/bdshs`.  \n" + SETTINGS + "\n\nAbout **4 hours**.",
        ["bdshs"], [["bdshs", m, 42, "full"] for m in ALL5], "rev14a"),
    "nb14b_revision_bdshs_seeds.ipynb": training_nb(
        "REVISION 14b - BD-SHS, seeds 43 and 44, all five models",
        "# Revision 14b: two more seeds for every model (BD-SHS)\n\n"
        "Reviewer 1 asked for multiple seeds for all fine-tuned models. Together with 14a (seed 42)\n"
        "this gives three seeds per model under the identical Table III protocol.\n\n"
        "**Inputs:** dataset `naurosromim/bdshs`.  \n" + SETTINGS + "\n\nAbout **6-7 hours**; it can run at the same time as 14a.",
        ["bdshs"], [["bdshs", m, s, "seed"] for s in (43, 44) for m in ALL5], "rev14b"),
    "nb14c_revision_dataset2.ipynb": training_nb(
        "REVISION 14c - second dataset (BLP-2025 shared task 1A)",
        "# Revision 14c: an independent second Bangla hate-speech dataset\n\n"
        "Both reviewers asked for more data. Same five models and protocol on the BLP-2025 shared task\n"
        "subtask 1A (Bangla YouTube comments, hate = any label other than None), official train and test\n"
        "splits. Only 10 of its 45,691 comments also occur in BD-SHS. The notebook downloads it from\n"
        "GitHub, so it needs no dataset input.\n\n"
        "**Inputs:** none.  \n" + SETTINGS + "\n\nAbout **4 hours**.",
        ["blp25"], [["blp25", m, 42, "full"] for m in ALL5], "rev14c"),
    **{f"nb14{nb}_revision_blp_seed{s}.ipynb": training_nb(
        f"REVISION 14{nb} - BLP-2025, seed {s}, all five models",
        f"# Revision 14{nb}: seed {s} on the second dataset (BLP-2025)\n\n"
        "Reviewer 1 asked for multiple seeds for all fine-tuned models. 14b does this for BD-SHS; this\n"
        f"notebook adds seed {s} on BLP-2025 (14c is seed 42), same protocol, ZWNJ and CGJ attacks.\n"
        "14e (seed 43) and 14f (seed 44) are independent and can run at the same time.\n\n"
        "**Inputs:** none (downloads BLP-2025 from GitHub).  \n" + SETTINGS + "\n\nAbout **2 hours**.",
        ["blp25"], [["blp25", m, s, "seed"] for m in ALL5], f"rev14{nb}")
       for nb, s in (("e", 43), ("f", 44))},
    "nb14d_revision_advtrain.ipynb": training_nb(
        "REVISION 14d - adversarial training vs characters it never saw",
        "# Revision 14d: adversarial training, and whether it transfers\n\n"
        "Reviewer 1 asked to compare our preprocessing defense with adversarial training. XLM-R is\n"
        "trained with half its training comments repeated under the verified ZWNJ attack (as in Step 9),\n"
        "then attacked with ZWNJ and with characters it never saw (CGJ, VS16, LRM). The plain XLM-R\n"
        "baseline for comparison is seed 42 in 14a.\n\n"
        "**Inputs:** dataset `naurosromim/bdshs`.  \n" + SETTINGS + "\n\nAbout **2 hours** (one GPU is enough).",
        ["bdshs"], [["bdshs", "xlmr", 42, "advtrain"]], "rev14d"),
}

LLM_INSTALL = '''!pip install -q "transformers==4.46.3" "tokenizers<0.21" accelerate bitsandbytes "emoji==1.7.0" regex ftfy uharfbuzz sentencepiece protobuf
!pip install -q git+https://github.com/csebuetnlp/normalizer
import transformers, torch
print("transformers", transformers.__version__, "| GPUs", torch.cuda.device_count())'''

LLM_SETUP = '''import sys, os, glob, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/kaggle/working")
import render_attack as ra
import numpy as np, pandas as pd, torch
from normalizer import normalize
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
R = ra.Renderer(ra.fetch_fonts(f"{OUT}/fonts"))
DI = ra.is_default_ignorable

# Same sample as Step 7 (Qwen2.5-7B in the submitted paper)
paths = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True) if "/bdshs/" in p]
raw = pd.read_csv([p for p in paths if "test" in os.path.basename(p).lower()][0])
obj = [c for c in raw.columns if raw[c].dtype == object or pd.api.types.is_string_dtype(raw[c])]
TEXT = max(obj, key=lambda c: raw[c].astype(str).str.len().mean())
LABEL = [c for c in raw.columns if c != TEXT and raw[c].dropna().nunique() == 2][0]
test = raw[[TEXT, LABEL]].dropna().copy(); test.columns = ["text", "label"]
test["text"] = test["text"].astype(str); test["label"] = test["label"].astype(int)
test = test[test.text.str.strip().str.len() > 0].reset_index(drop=True)
N = 300
hate = test[test.label == 1].sample(N, random_state=0).text.tolist()
ben = test[test.label != 1].sample(N, random_state=0).text.tolist()
ATK = {"zwnj": [R.attack(t, "ALL", mode="inert", seed=i)[0] for i, t in enumerate(hate)],
       "cgj": [R.attack(t, "ALL", char=ra.CGJ, mode="inert", seed=i)[0] for i, t in enumerate(hate)]}
BEN_ATK = {"zwnj": [R.attack(t, "ALL", mode="both", seed=i)[0] for i, t in enumerate(ben)],
           "cgj": [R.attack(t, "ALL", char=ra.CGJ, mode="inert", seed=i)[0] for i, t in enumerate(ben)]}
print("data ok | hate", len(hate), "| benign", len(ben))

def find_local(*words):
    """A model added with Add Input > Models lives under /kaggle/input; find its folder."""
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        p = os.path.dirname(cfg).lower()
        if all(w in p for w in words):
            return os.path.dirname(cfg)
    return None

LLMS = [  # name, path or hub id, 4-bit
    ("Qwen2.5-7B-Instruct", find_local("qwen2.5", "7b") or "Qwen/Qwen2.5-7B-Instruct", False),
    # Kaggle has paused Llama access requests; unsloth hosts the same weights ungated
    # (Llama 3.1 Community License and Gemma Terms of Use still apply).
    ("Llama-3.1-8B-Instruct", find_local("llama-3.1", "8b") or "unsloth/Llama-3.1-8B-Instruct", False),
    ("Gemma-2-9B-it", find_local("gemma-2", "9b") or "unsloth/gemma-2-9b-it", False),
    ("Qwen2.5-14B-Instruct", find_local("qwen2.5", "14b") or "Qwen/Qwen2.5-14B-Instruct", True),
]
LLMS = [x for x in LLMS if ONLY is None or x[0] in ONLY]
for n, p, q in LLMS: print(f"{n:24s} -> {p}")'''

LLM_RUN = '''from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
SYS = ("You are a strict content moderator for Bengali text. Does the following text contain "
       "hate speech, abuse, or toxicity? Answer strictly with 'YES' or 'NO' and nothing else.")

def load(path, four_bit):
    tk = AutoTokenizer.from_pretrained(path); tk.padding_side = "left"
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    kw = {"device_map": "auto"}
    if "gemma" in path.lower():
        kw["attn_implementation"] = "eager"   # Gemma 2 logit soft-capping needs eager attention
        kw["max_memory"] = {0: "11GiB", 1: "13GiB"}   # spread it: GPU 0 also holds the activations
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = torch.float16
    return tk, AutoModelForCausalLM.from_pretrained(path, **kw).eval()

def prompt(tk, t):
    msg = f"Text: {normalize(t)}"
    try:
        return tk.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:   # Gemma has no system role: put the instruction in the user turn
        return tk.apply_chat_template([{"role": "user", "content": SYS + "\\n\\n" + msg}],
                                      tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def ask(tk, mdl, texts, bs=None):
    bs = bs or BS
    ans = []
    for s in range(0, len(texts), bs):
        enc = tk([prompt(tk, t) for t in texts[s:s + bs]], return_tensors="pt", padding=True,
                 add_special_tokens=False).to(mdl.device)
        out = mdl.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tk.pad_token_id)
        for o in tk.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            o = o.strip().upper()
            ans.append("YES" if o.startswith("YES") else "NO" if o.startswith("NO") else "OTHER")
    return np.array(ans)

EV, CTRL = [], []
for name, path, q in LLMS:
    if path is None:
        print(f"SKIP {name}: add it with Add Input > Models (accept the licence first)"); continue
    # Gemma 2's 256k-word output layer runs out of T4 memory at batch 8
    BS = 2 if "gemma" in name.lower() else 8
    try:
        tk, mdl = load(path, q)
        ch, cb = ask(tk, mdl, hate), ask(tk, mdl, ben)
        tgt = np.where(ch == "YES")[0]
        rep = ask(tk, mdl, [hate[i] for i in tgt])
        print(f"{name}: clean hate recall {len(tgt)}/{N}")
        for a, texts in ATK.items():
            at = [texts[i] for i in tgt]
            r_atk = ask(tk, mdl, at)
            r_old = ask(tk, mdl, [R.defend_render_equivalence(x) for x in at])
            r_new = ask(tk, mdl, [R.defend_render_equivalence(x, invisible=DI) for x in at])
            EV.append({"model": name, "attack": a, "n_targets": len(tgt),
                       "evaded": int((r_atk != "YES").sum()), "evasion": round(float((r_atk != "YES").mean()), 4),
                       "unparseable": round(float((r_atk == "OTHER").mean()), 4),
                       "evasion_after_old_defense": round(float((r_old != "YES").mean()), 4),
                       "evasion_after_generalized_defense": round(float((r_new != "YES").mean()), 4)})
            ba = ask(tk, mdl, BEN_ATK[a])
            CTRL.append({"model": name, "attack": a, "hate_recall_clean": round(len(tgt) / N, 4),
                         "benign_YES_clean": round(float((cb == "YES").mean()), 4),
                         "benign_YES_attacked": round(float((ba == "YES").mean()), 4),
                         "benign_answer_flip": round(float((cb != ba).mean()), 4),
                         "clean_repeat_disagreement": round(float((rep != "YES").mean()), 4)})
            print("  ", EV[-1])
        pd.DataFrame(EV).to_csv(f"{OUT}/llm_evasion.csv", index=False)
        pd.DataFrame(CTRL).to_csv(f"{OUT}/llm_controls.csv", index=False)
    except Exception as ex:
        print(f"FAILED {name}: {type(ex).__name__}: {str(ex)[:300]}")
    finally:
        mdl = None; gc.collect(); torch.cuda.empty_cache()'''

LLM_SUMMARY = '''import shutil
pd.set_option("display.width", 250)
print("=" * 100); print("REVISION 15 - LLM moderators, BD-SHS test sample (300 hate + 300 benign)"); print("=" * 100)
print(pd.DataFrame(EV).to_string(index=False)); print(); print(pd.DataFrame(CTRL).to_string(index=False))
shutil.make_archive(f"/kaggle/working/{ZIPNAME}", "zip", OUT)
print(f"\\nDownload /kaggle/working/{ZIPNAME}.zip from the Output tab and send it to me."); print("=" * 100)'''

def llm_nb(only, zipname, intro, minutes):
    cfg = f"ONLY = {only!r}   # None = all four models\nZIPNAME = {zipname!r}"
    return notebook([
        md(intro + "\n\n**Inputs:** dataset `naurosromim/bdshs` only. Models download from Hugging Face\n"
           "(Llama and Gemma from the ungated unsloth copies). A model that fails to load is skipped.  \n"
           + SETTINGS + f"\n\nAbout **{minutes}**."),
        code(LLM_INSTALL),
        code("%%writefile /kaggle/working/render_attack.py\n" + SRC["render_attack.py"]),
        code(cfg), code(LLM_SETUP), code(LLM_RUN), code(LLM_SUMMARY)])


NBS["nb15_revision_llms.ipynb"] = llm_nb(
    None, "rev15",
    "# Revision 15: more LLM moderators\n\n"
    "Reviewer 1 asked for more and stronger LLMs than one 7B model. Same prompt, sample and attack as\n"
    "Step 7, plus the CGJ attack and the generalized defense, on four instruction models.", "2 hours")
NBS["nb15b_revision_gemma.ipynb"] = llm_nb(
    ["Gemma-2-9B-it"], "rev15b",
    "# Revision 15b: Gemma-2-9B only\n\n"
    "Rerun of the one model that ran out of GPU memory in nb15, now with batch size 2 and the model\n"
    "spread across both GPUs. Everything else is identical to nb15.", "45 minutes")

for name, nb in NBS.items():
    p = os.path.join(ROOT, "notebooks", name)
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", os.path.relpath(p, ROOT))

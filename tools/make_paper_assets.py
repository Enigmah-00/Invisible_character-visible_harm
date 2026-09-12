#!/usr/bin/env python3
"""
Build every figure and table in the paper from result CSVs.

Rule: nothing is drawn or typed by hand. Each asset names the CSV and columns it
came from. A missing CSV is a hard error, so the paper cannot contain a number
that no notebook produced.

Usage:  python tools/make_paper_assets.py
Inputs: results/*.csv  (download from the Kaggle committed-version Output tab)
Output: paper/fig/*.pdf, paper/tab/*.tex, paper/NUMBERS.md
"""
import os
import sys
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG, TAB = "results", "paper/fig", "paper/tab"

REQUIRED = {
    "step6_ladder.csv":          "nb06 cell 8",
    "step6_defense.csv":         "nb06 cell 8",
    "step6_attack_meta.csv":     "nb06 cell 5",
    "step6_crossfont.csv":       "nb06 cell 5",
    "step6_wild_collateral.csv": "nb06 cell 6",
    "step6_latency.csv":         "nb06 cell 6",
    "step6_exact_recovery.csv":  "nb06 cell 6",
    "step6_clean_cost.csv":      "nb06 cell 8",
    "step6_benign.csv":          "nb06 cell 8",
    "step6c_ladder.csv":         "nb06c cell 6",
    "step6c_defense.csv":        "nb06c cell 6",
    "step6c_residual_flag.csv":  "nb06c cell 4",
    "step6c_clean.csv":          "nb06c cell 6",
    "step6b_ladder.csv":         "nb06b cell 6",
    "step6d_pipeline.csv":       "nb06d cell 6",
    "step7_llm_evasion.csv":     "nb07 cell 5",
    "step7_llm_controls.csv":    "nb07 cell 5",
    "tokenizer_survey.csv":      "nb02 cell 3",
}

missing = [f for f in REQUIRED if not os.path.exists(os.path.join(RES, f))]
if missing:
    print("MISSING result files. Download them before building assets:\n")
    for f in missing:
        print(f"  results/{f:<28} produced by {REQUIRED[f]}")
    print("\nKaggle: open the committed version -> Output tab -> Download all,")
    print("then unzip the CSVs into results/ . Nothing is invented here.")
    sys.exit(1)

L = lambda f: pd.read_csv(os.path.join(RES, f))
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL1 = (3.4, 2.1)     # IEEE single column
COL2 = (7.16, 2.3)    # IEEE double column
notes = []


def save(fig, name, caption_src):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    notes.append(f"- `{name}` <- {caption_src}")
    print("wrote", p)


# ---------------------------------------------------------------- figure 1
ladder = L("step6_ladder.csv")
meta = L("step6_attack_meta.csv").set_index("config")["mean_inserted"]
order = ["gated_inert_b1", "gated_inert_b3", "gated_inert_b5",
         "gated_inert_b10", "gated_inert_ALL"]
fig, ax = plt.subplots(figsize=COL1)
for model, style in [("xlm-roberta-base", dict(marker="o", color="#B4413C")),
                     ("csebuetnlp/banglabert", dict(marker="s", color="#1F4E79"))]:
    d = ladder[ladder.model == model]
    g = d[d.config.isin(order)].groupby("config")["evasion"].agg(["mean", "std"])
    g = g.reindex(order)
    x = [meta[c] for c in order]
    ax.errorbar(x, g["mean"] * 100, yerr=(g["std"] * 100).fillna(0),
                capsize=2, lw=1.2, ms=3.5,
                label=model.split("/")[-1], **style)
ax.set_xscale("log")
ax.set_xlabel("invisible characters inserted per comment")
ax.set_ylabel("evasion rate (\\%)" if False else "evasion rate (%)")
ax.set_ylim(-3, 100)
ax.legend(loc="upper left", frameon=False)
save(fig, "fig_ladder.pdf", "step6_ladder.csv (evasion, mean/std over seeds) x step6_attack_meta.csv (mean_inserted)")

# ---------------------------------------------------------------- figure 2
dfn = L("step6_defense.csv")
coll = L("step6_wild_collateral.csv")
d = dfn[(dfn.model == "xlm-roberta-base") & (dfn.attack == "gated_inert_ALL")]
d = d.groupby("defense")["evasion"].mean() * 100
names = ["none", "nfc", "strip_all", "position_aware", "render_equivalence"]
lab = ["none", "NFC", "strip-all", "position-\naware", "render-\nequivalence"]
c2 = coll.set_index("defense")["rendering_changed_any_font"].reindex(names).fillna(0)
tot = int(coll["docs_with_zw"].iloc[0])
fig, axes = plt.subplots(1, 2, figsize=COL2)
axes[0].bar(lab, [d.get(n, float("nan")) for n in names], color="#B4413C", width=.6)
axes[0].set_ylabel("evasion rate (%)")
axes[0].set_title("(a) security: attack stopped?")
axes[1].bar(lab, c2.values, color="#1F4E79", width=.6)
axes[1].set_ylabel(f"real comments visibly corrupted\n(of {tot} containing zero-width)")
axes[1].set_title("(b) collateral: real text broken?")
for a in axes:
    a.tick_params(axis="x", labelsize=6.5)
save(fig, "fig_defense.pdf", "step6_defense.csv (evasion) + step6_wild_collateral.csv (rendering_changed_any_font)")

# ---------------------------------------------------------------- figure 3
# Only built from step6b_inflation.csv, which uses the rendering-verified attack.
# step5_llm_tokenizers.csv used the old ungated attack (visible in 73.3% of
# comments) and must not appear in the paper.
if os.path.exists(os.path.join(RES, "step6b_inflation.csv")):
    tok = L("step6b_inflation.csv").sort_values("inflation_pct")
    fig, ax = plt.subplots(figsize=COL1)
    y = range(len(tok))
    ax.barh([i + .2 for i in y], tok["clean_tokens"], height=.38, label="clean", color="#1F4E79")
    ax.barh([i - .2 for i in y], tok["attacked_tokens"], height=.38, label="attacked", color="#B4413C")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r.tokenizer}\n({r.family})" for r in tok.itertuples()], fontsize=6)
    ax.set_xlabel("mean tokens per comment")
    ax.legend(frameon=False)
    save(fig, "fig_inflation.pdf", "step6b_inflation.csv (clean_tokens, attacked_tokens)")
else:
    print("SKIP fig_inflation.pdf - run nb06b to produce results/step6b_inflation.csv.")
    print("     step5_llm_tokenizers.csv is NOT usable: it measured the old ungated attack.")

# ---------------------------------------------------------------- figure 4
cf = L("step6_crossfont.csv").sort_values("visibly_broken", ascending=True)
fig, ax = plt.subplots(figsize=COL1)
ax.barh(cf["held_out_font"], cf["visibly_broken"] / cf["of"] * 100, color="#B4413C")
ax.set_xlabel("comments visibly broken (%)")
save(fig, "fig_crossfont.pdf", "step6_crossfont.csv (visibly_broken / of)")


# ----------------------------------------------------------------- tables
def tabular(df, cols, header, caption, label, fmt=None, width=None):
    fmt = fmt or {}
    body = []
    for _, r in df.iterrows():
        cells = [fmt.get(c, str)(r[c]) for c in cols]
        body.append(" & ".join(cells) + r" \\")
    spec = width or ("l" + "r" * (len(cols) - 1))
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\begin{tabular}{" + spec + "}", r"\hline",
        " & ".join(header) + r" \\", r"\hline",
        *body, r"\hline", r"\end{tabular}", r"\end{table}", ""])


def w(name, text, src):
    p = os.path.join(TAB, name)
    open(p, "w").write(text)
    notes.append(f"- `{name}` <- {src}")
    print("wrote", p)


pct = lambda v: f"{float(v)*100:.1f}"
esc = lambda v: str(v).replace("_", r"\_")

t = L("tokenizer_survey.csv")
t = t[t.tokenizer != "LOAD_FAIL"].copy()
t["vuln"] = t["vulnerable"].map(lambda v: r"\textbf{yes}" if str(v).lower() == "true" else "no")
t["infl"] = ((t["atk_len"] / t["clean_len"] - 1) * 100).round(0).astype(int).astype(str) + r"\%"
w("tab_tokenizer.tex", tabular(
    t, ["model", "tokenizer", "vuln", "infl"],
    ["Model", "Tokenizer class", "Exposed", "Token infl."],
    "Zero-width characters reach the model only for SentencePiece and BPE tokenizers. "
    "\\texttt{BertTokenizer.\\_clean\\_text} deletes every Unicode category-C character.",
    "tokenizer", {"model": esc, "tokenizer": esc}, "llcr"),
    "tokenizer_survey.csv")

lad = ladder.groupby(["model", "config"], sort=False).agg(
    ev=("evasion", "mean"), sd=("evasion", "std"),
    ev128=("evasion_at_maxlen128", "mean")).reset_index()
lad = lad[lad.config.isin(order + ["gated_adaptive_hasant_ALL", "ungated_semi_ALL_old"])]
# BanglaBERT is 0.0 at every budget; one row says it, fourteen rows waste a column.
bb = lad[lad.model == "csebuetnlp/banglabert"]
lad = pd.concat([lad[lad.model != "csebuetnlp/banglabert"],
                 bb[bb.config == "gated_inert_ALL"].assign(config="all budgets")])
lad["ins"] = lad["config"].map(meta).round(2)
lad["vis"] = lad["config"].map(L("step6_attack_meta.csv").set_index("config")["pct_visible_any_font"])
lad["evs"] = lad.apply(lambda r: f"{r.ev*100:.1f}" + (f" $\\pm$ {r.sd*100:.1f}" if pd.notna(r.sd) else ""), axis=1)
w("tab_ladder.tex", tabular(
    lad, ["model", "config", "ins", "vis", "evs", "ev128"],
    ["Model", "Attack", "Ins.", "Visible", "Evasion (\\%)", "@128"],
    "Evasion under the rendering-verified attack, mean $\\pm$ std over 3 seeds. "
    "`Visible' is the share of comments that render differently in any of 8 fonts. "
    "The @128 column shows the truncation control.",
    "ladder", {"model": esc, "config": esc, "vis": pct, "ev128": pct}, "llrrrr"),
    "step6_ladder.csv + step6_attack_meta.csv")

lat = L("step6_latency.csv").set_index("defense")["ms_per_comment"]
dd = dfn[dfn.model == "xlm-roberta-base"].groupby(["defense", "attack"])["evasion"].mean().unstack()
dd = dd.reindex(names)
c = coll.set_index("defense")
rows = []
for n in names:
    rows.append({
        "defense": n,
        "inert": dd.loc[n, "gated_inert_ALL"] if "gated_inert_ALL" in dd else float("nan"),
        "adapt": dd.loc[n, "gated_adaptive_hasant_ALL"] if "gated_adaptive_hasant_ALL" in dd else float("nan"),
        "vis": c["rendering_changed_any_font"].get(n, 0),
        "ms": lat.get(n, float("nan"))})
w("tab_defense.tex", tabular(
    pd.DataFrame(rows), ["defense", "inert", "adapt", "vis", "ms"],
    ["Defense", "Evasion", "Adaptive", f"Corrupted/{tot}", "ms"],
    "All defenses stop the attack. They differ on collateral damage to the "
    f"{tot} real comments that already contain zero-width characters.",
    "defense", {"defense": esc, "inert": pct, "adapt": pct,
                "vis": lambda v: str(int(v)), "ms": lambda v: f"{float(v):.3f}"}, "lrrrr"),
    "step6_defense.csv + step6_wild_collateral.csv + step6_latency.csv")

# ---- cross-model table: tokenizer family decides exposure ----------------
rows = []
for f, src in [("step6_ladder.csv", "nb06"), ("step6b_ladder.csv", "nb06b"), ("step6c_ladder.csv", "nb06c")]:
    d = L(f)
    d = d[d.config == "gated_inert_ALL"]
    for m, g in d.groupby("model", sort=False):
        rows.append({"model": m, "tok": g["tokenizer"].iloc[0],
                     "ev": g["evasion"].mean(), "n": int(g["n_targets"].iloc[0])})
cm = pd.DataFrame(rows).drop_duplicates("model")
FAM = {"BertTokenizer": "WordPiece", "XLMRobertaTokenizer": "SentencePiece",
       "T5Tokenizer": "SentencePiece", "DebertaV2Tokenizer": "SentencePiece"}
cm["fam"] = cm["tok"].map(lambda t: FAM.get(t, t))
acc = {}
for f, col in [("step6_clean_cost.csv", "defense_on_clean"), ("step6c_clean.csv", "defense_on_clean")]:
    d = L(f); d = d[d[col] == "none"]
    for m, g in d.groupby("model"):
        acc[m] = g["acc"].mean()
cm["acc"] = cm["model"].map(acc)
cm = cm.sort_values("ev")
w("tab_crossmodel.tex", tabular(
    cm, ["model", "fam", "acc", "ev"],
    ["Model", "Tokenizer family", "Clean acc.", "Evasion (\\%)"],
    "Exposure follows the tokenizer family, not the model or the language. "
    "BanglaBERT and BanglaT5 come from the same group and the same Bangla corpus; "
    "only the tokenizer differs.",
    "crossmodel", {"model": esc, "fam": esc,
                   "acc": lambda v: f"{float(v):.3f}" if pd.notna(v) else "--",
                   "ev": pct}, "llrr"),
    "step6_ladder.csv + step6b_ladder.csv + step6c_ladder.csv + clean_cost")

# ---- residual flag table --------------------------------------------------
fl = L("step6c_residual_flag.csv")
w("tab_flag.tex", tabular(
    fl, ["set", "kind", "n", "flag_rate"],
    ["Input set", "Type", "$n$", "Flagged (\\%)"],
    "Residual flag after render-equivalence. Imperceptible attacks are already "
    "removed, so what remains and fires the flag is a manipulation a reader can see.",
    "flag", {"set": esc, "kind": esc, "n": lambda v: str(int(v)), "flag_rate": pct}, "llrr"),
    "step6c_residual_flag.csv")

# ---- end-to-end pipeline table -------------------------------------------
pl = L("step6d_pipeline.csv")
w("tab_pipeline.tex", tabular(
    pl, ["attack", "evasion_no_defense", "evasion_after_render_eq", "flag_rate", "UNMITIGATED_evasion"],
    ["Attack", "No defense", "After ours", "Flagged", "Unmitigated"],
    "End-to-end pipeline on XLM-R. `Unmitigated\' means the comment evaded the "
    "classifier and the flag never fired, so no reviewer would ever see it. "
    "Imperceptible attacks are removed; visible ones are flagged.",
    "pipeline", {"attack": esc, "evasion_no_defense": pct,
                 "evasion_after_render_eq": pct, "flag_rate": pct,
                 "UNMITIGATED_evasion": pct}, "lrrrr"),
    "step6d_pipeline.csv")

llm = L("step7_llm_evasion.csv")
w("tab_llm.tex", tabular(
    llm, ["attack", "n_targets", "pct_visible_any_font", "evasion_NO", "YES_after_render_equivalence"],
    ["Attack", "$n$", "Visible", "Evasion", "Restored"],
    "Qwen2.5-7B-Instruct as a zero-shot Bangla moderator, BD-SHS test split.",
    "llm", {"attack": esc, "n_targets": lambda v: str(int(v)),
            "pct_visible_any_font": pct, "evasion_NO": pct,
            "YES_after_render_equivalence": pct}, "lrrrr"),
    "step7_llm_evasion.csv")

open("paper/NUMBERS.md", "w").write(
    "# Provenance\n\nEvery asset below is generated by `tools/make_paper_assets.py` "
    "from a CSV written by a notebook. No value is entered by hand.\n\n" +
    "\n".join(notes) + "\n")
print("\nwrote paper/NUMBERS.md")
print("done:", len([n for n in notes]), "assets")

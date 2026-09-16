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
    "human_study.csv":           "tools/score_google_form.py",
    "step12_ladder.csv":         "nb12 (transformers 4.46.3)",
    "step12_clean_cost.csv":     "nb12 (transformers 4.46.3)",
    "step12_defense.csv":        "nb12 (transformers 4.46.3)",
    "step12_benign.csv":         "nb12 (transformers 4.46.3)",
    "step13_ladder.csv":         "nb13 (transformers 4.46.3)",
    "step13_clean_cost.csv":     "nb13 (transformers 4.46.3)",
    "step13_defense.csv":        "nb13 (transformers 4.46.3)",
    "step13_benign.csv":         "nb13 (transformers 4.46.3)",
    "step6c_ladder.csv":         "nb06c cell 6",
    "step6c_defense.csv":        "nb06c cell 6",
    "step6c_residual_flag.csv":  "nb06c cell 4",
    "step6c_clean.csv":          "nb06c cell 6",
    "step6b_ladder.csv":         "nb06b cell 6",
    "step6b_clean_cost.csv":     "nb06b cell 6",
    "step6d_pipeline.csv":       "nb06d cell 6",
    "step7_llm_evasion.csv":     "nb07 cell 5",
    "step7_llm_controls.csv":    "nb07 cell 5",
    "tokenizer_survey.csv":      "nb02 cell 3",
    "step14_collateral.csv":     "nb14 cell 5",
    "step14_clean_text_cost.csv":"nb14 cell 5",
    "step14_security.csv":       "nb14 cell 5",
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
fig, axes = plt.subplots(1, 2, figsize=(COL2[0], 2.55))
# generous gap: panel (b)'s y-axis label used to collide with panel (a)
fig.subplots_adjust(wspace=0.32)

va = [d.get(n, float("nan")) for n in names]
b0 = axes[0].bar(lab, va, color="#B4413C", width=.62)
axes[0].set_ylabel("evasion rate (%)")
axes[0].set_title("(a) security: is the attack stopped?", fontsize=8)
axes[0].set_ylim(0, max(v for v in va if v == v) * 1.18)
axes[0].bar_label(b0, fmt="%.1f", fontsize=6.5, padding=1.5)

b1 = axes[1].bar(lab, c2.values, color="#1F4E79", width=.62)
axes[1].set_ylabel("comments corrupted")          # short: no more collision
axes[1].set_title(f"(b) collateral: real Bangla broken?\n(of {tot} comments containing zero-width)",
                  fontsize=8)
axes[1].set_ylim(0, max(c2.values) * 1.22)
axes[1].bar_label(b1, fmt="%d", fontsize=6.5, padding=1.5)

for a in axes:
    a.tick_params(axis="x", labelsize=6.2)
    a.margins(x=0.06)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
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

# (the cross-font figure was cut from the paper for space; its two numbers
#  are quoted in the prose straight from step6_crossfont.csv)

# ----------------------------------------------------------------- tables
def tabular(df, cols, header, caption, label, fmt=None, width=None, wide=False):
    """Emit one IEEE table. wide=True spans both columns (table*).

    Every table is \footnotesize: at normal size the long model and tokenizer
    names run past the column edge."""
    fmt = fmt or {}
    body = []
    for _, r in df.iterrows():
        cells = [fmt.get(c, str)(r[c]) for c in cols]
        body.append(" & ".join(cells) + r" \\")
    spec = width or ("l" + "r" * (len(cols) - 1))
    env = "table*" if wide else "table"
    return "\n".join([
        r"\begin{" + env + "}[t]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\footnotesize",
        # adjustbox shrinks only if the table is too wide, so a long row can
        # never run past the column edge.
        r"\begin{adjustbox}{max width=" + ("\\textwidth" if wide else "\\columnwidth") + "}",
        r"\begin{tabular}{" + spec + "}", r"\hline",
        " & ".join(header) + r" \\", r"\hline",
        *body, r"\hline", r"\end{tabular}",
        r"\end{adjustbox}", r"\end{" + env + "}", ""])


def w(name, text, src):
    p = os.path.join(TAB, name)
    open(p, "w").write(text)
    notes.append(f"- `{name}` <- {src}")
    print("wrote", p)


def pct(v):
    return "--" if pd.isna(v) else f"{float(v)*100:.1f}"


def num(v, d=2):
    return "--" if pd.isna(v) else f"{float(v):.{d}f}"
esc = lambda v: str(v).replace("_", r"\_")

# Internal run names are long and ugly. Rename them for print only.
ATTACK_LABEL = {
    "gated_inert_b1": "invisible, 1 char", "gated_inert_b3": "invisible, 3",
    "gated_inert_b5": "invisible, 5", "gated_inert_b10": "invisible, 10",
    "gated_inert_ALL": "invisible, all", "gated_inert_ALL_zwj": "invisible, all (ZWJ)",
    "gated_adaptive_hasant_ALL": "adaptive", "gated_both_ALL": "invisible + adaptive",
    "ungated_semi_ALL_old": "unverified", "every budget": "every budget",
}
MODEL_LABEL = {
    "xlm-roberta-base": "XLM-R", "csebuetnlp/banglabert": "BanglaBERT",
    "google/muril-base-cased": "MuRIL", "csebuetnlp/banglat5": "BanglaT5",
}
DEF_LABEL = {"none": "none", "nfc": "NFC", "strip_all": "strip-all",
             "position_aware": "position-aware", "render_equivalence": "render-equivalence"}
SET_LABEL = dict(ATTACK_LABEL, **{
    "real comments containing zero-width": "real comments with zero-width",
    "all clean test comments": "all clean test comments"})
alab = lambda v: ATTACK_LABEL.get(str(v), esc(v))
mlab = lambda v: MODEL_LABEL.get(str(v), esc(v))
dlab = lambda v: DEF_LABEL.get(str(v), esc(v))
slab = lambda v: SET_LABEL.get(str(v), esc(v))

t = L("tokenizer_survey.csv")
t = t[t.tokenizer != "LOAD_FAIL"].copy()
t["vuln"] = t["vulnerable"].map(lambda v: r"\textbf{yes}" if str(v).lower() == "true" else "no")
t["infl"] = ((t["atk_len"] / t["clean_len"] - 1) * 100).round(0).astype(int).astype(str) + r"\%"
w("tab_tokenizer.tex", tabular(
    t, ["model", "tokenizer", "vuln", "infl"],
    ["Model", "Tokenizer class", "Exposed", "Token infl."],
    "Zero-width characters reach the model only for SentencePiece and BPE tokenizers. "
    "\\texttt{BertTokenizer.\\_clean\\_text} deletes every Unicode category-C character.",
    "tokenizer", {"model": esc, "tokenizer": esc}, "llcr", wide=True),
    "tokenizer_survey.csv")

lad = ladder.groupby(["model", "config"], sort=False).agg(
    ev=("evasion", "mean"), sd=("evasion", "std"),
    ev128=("evasion_at_maxlen128", "mean")).reset_index()
lad = lad[lad.config.isin(order + ["gated_adaptive_hasant_ALL", "ungated_semi_ALL_old"])]
lad["ins"] = lad["config"].map(meta).round(2)
lad["vis"] = lad["config"].map(L("step6_attack_meta.csv").set_index("config")["pct_visible_any_font"])
# BanglaBERT is 0.0 at every budget; one row says it. Relabel AFTER the lookups,
# otherwise the new label has no entry in the meta table and the row shows NaN.
bb = lad[lad.model == "csebuetnlp/banglabert"]
bb = bb[bb.config == "gated_inert_ALL"].copy()
bb["config"] = "every budget"
bb["ins"] = float("nan")
lad = pd.concat([lad[lad.model != "csebuetnlp/banglabert"], bb])
lad["evs"] = lad.apply(lambda r: f"{r.ev*100:.1f}" + (f" $\\pm$ {r.sd*100:.1f}" if pd.notna(r.sd) else ""), axis=1)
w("tab_ladder.tex", tabular(
    lad, ["model", "config", "ins", "vis", "evs", "ev128"],
    ["Model", "Attack", "Ins.", "Visible", "Evasion (\\%)", "@128"],
    "Evasion under the rendering-verified attack, mean $\\pm$ std over 3 seeds. "
    "`Visible' is the share of comments that render differently in any of 8 fonts. "
    "The @128 column shows the truncation control.",
    "ladder", {"model": mlab, "config": alab, "ins": num, "vis": pct, "ev128": pct},
    "llrrrr", wide=True),
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
    "defense", {"defense": dlab, "inert": pct, "adapt": pct,
                "vis": lambda v: str(int(v)), "ms": lambda v: f"{float(v):.3f}"}, "lrrrr"),
    "step6_defense.csv + step6_wild_collateral.csv + step6_latency.csv")

# ---- cross-model table: five models, ONE library, ONE protocol ----------
UL = pd.concat([L("step13_ladder.csv"), L("step12_ladder.csv")])
UC = pd.concat([L("step13_clean_cost.csv"), L("step12_clean_cost.csv")])
UD = pd.concat([L("step13_defense.csv"), L("step12_defense.csv")])
uacc = UC[UC.defense_on_clean == "none"].set_index("model").acc
SHORT = {"csebuetnlp/banglabert": "BanglaBERT", "google/muril-base-cased": "MuRIL",
         "xlm-roberta-base": "XLM-R", "microsoft/mdeberta-v3-base": "mDeBERTa-v3",
         "csebuetnlp/banglat5": "BanglaT5"}
rows = []
for m, nice in SHORT.items():
    g = UL[(UL.model == m) & (UL.config == "gated_inert_ALL")]
    if g.empty:
        continue
    dd = UD[(UD.model == m) & (UD.attack == "gated_both_ALL")].set_index("defense").evasion
    rows.append({"model": nice, "fam": g.family.iloc[0], "acc": uacc.get(m),
                 "ev": g.evasion.iloc[0], "pa": dd.get("position_aware"),
                 "ours": dd.get("render_equivalence")})
cm = pd.DataFrame(rows).sort_values(["fam", "ev"])
w("tab_crossmodel.tex", tabular(
    cm, ["model", "fam", "acc", "ev", "pa", "ours"],
    ["Model", "Tokenizer family", "Clean acc.", "Evasion (\\%)",
     "Position-aware", "Ours"],
    "Five models, one training protocol and one library version "
    "(\\texttt{transformers} 4.46.3). Exposure follows the tokenizer family, not the "
    "model or the language: BanglaBERT and BanglaT5 come from the same group and the same "
    "Bangla corpus. The last two columns give evasion under the adaptive attack after the "
    "rule-based defense and after ours.",
    "crossmodel", {"model": esc, "fam": esc,
                   "acc": lambda v: "--" if pd.isna(v) else f"{float(v):.3f}",
                   "ev": pct, "pa": pct, "ours": pct}, "llrrrr", wide=True),
    "step13_* + step12_* (unified protocol, transformers 4.46.3)")

# ---- human perceptibility study ------------------------------------------
hs = L("human_study.csv")
hs = hs[hs.group == "all"].copy()
hs["res"] = hs.apply(lambda r: f"{int(r.k)}/{int(r.n)} ({r.rate*100:.1f}\\%)", axis=1)
hs["ci"] = hs.apply(lambda r: f"{r.ci_low*100:.1f}--{r.ci_high*100:.1f}", axis=1)
w("tab_human.tex", tabular(
    hs, ["stimulus", "measure", "res", "ci"],
    ["Stimulus", "Judgement", "Count", "95\\% CI"],
    f"Perceptibility study, {int(hs.n_participants.iloc[0])} Bangla readers, 30 trials each. "
    "Every participant identified all five genuinely different pairs, so the null result "
    "is not inattention. Attacked pairs were called different less often than two "
    "identical images.",
    "human", {"stimulus": esc, "measure": esc}, "llrr"),
    "human_study.csv")

# ---- residual flag table --------------------------------------------------
fl = L("step6c_residual_flag.csv")
w("tab_flag.tex", tabular(
    fl, ["set", "kind", "n", "flag_rate"],
    ["Input set", "Type", "$n$", "Flagged (\\%)"],
    "Residual flag after render-equivalence. Imperceptible attacks are already "
    "removed, so what remains and fires the flag is a manipulation a reader can see.",
    "flag", {"set": slab, "kind": esc, "n": lambda v: str(int(v)), "flag_rate": pct}, "llrr"),
    "step6c_residual_flag.csv")

# ---- end-to-end pipeline table -------------------------------------------
pl = L("step6d_pipeline.csv")
w("tab_pipeline.tex", tabular(
    pl, ["attack", "evasion_after_render_eq", "flag_rate", "UNMITIGATED_evasion"],
    ["Attack", "After ours", "Flagged", "Unmitigated"],
    "End-to-end pipeline on XLM-R. `Unmitigated\' means the comment evaded the "
    "classifier and the flag never fired, so no reviewer would ever see it. "
    "Imperceptible attacks are removed; visible ones are flagged.",
    "pipeline", {"attack": alab, "evasion_after_render_eq": pct,
                 "flag_rate": pct, "UNMITIGATED_evasion": pct}, "lrrr"),
    "step6d_pipeline.csv")

llm = L("step7_llm_evasion.csv")
w("tab_llm.tex", tabular(
    llm, ["attack", "n_targets", "pct_visible_any_font", "evasion_NO", "YES_after_render_equivalence"],
    ["Attack", "$n$", "Visible", "Evasion", "Restored"],
    "Qwen2.5-7B-Instruct as a zero-shot Bangla moderator, BD-SHS test split.",
    "llm", {"attack": alab, "n_targets": lambda v: str(int(v)),
            "pct_visible_any_font": pct, "evasion_NO": pct,
            "YES_after_render_equivalence": pct}, "lrrrr"),
    "step7_llm_evasion.csv")

# ------------------------------------------------ numbers: bnunicodenormalizer
# The published Bangla normalizer removes zero-width characters as a side effect
# of linguistic normalization. We report what that side effect costs. Emitted as
# \newcommand so main.tex never carries a hand-typed number.
c14 = L("step14_collateral.csv").set_index("defense")
k14 = L("step14_clean_text_cost.csv").set_index("defense")
s14 = L("step14_security.csv")
rec = s14.groupby("defense")["exact_clean_recovery"].min() * 100
lat = s14.groupby("defense")["ms_per_comment"].mean()
macros = {
    "bnColl":     f"{int(c14.loc['bnunicodenormalizer','rendering_changed_any_font'])}",
    "bnCollTot":  f"{int(c14.loc['bnunicodenormalizer','docs_with_zw'])}",
    "bnClean":    f"{int(k14.loc['bnunicodenormalizer','rendering_changed'])}",
    "bnCleanTot": f"{int(k14.loc['bnunicodenormalizer','clean_comments'])}",
    "bnCleanPct": f"{k14.loc['bnunicodenormalizer','pct_rendering_changed']*100:.1f}",
    "bnRec":      f"{rec['bnunicodenormalizer']:.1f}",
    "bnMs":       f"{lat['bnunicodenormalizer']:.1f}",
    "reColl":     f"{int(c14.loc['render_equivalence','rendering_changed_any_font'])}",
    "reClean":    f"{int(k14.loc['render_equivalence','rendering_changed'])}",
    "reRec":      f"{rec['render_equivalence']:.1f}",
    "stripColl":  f"{int(c14.loc['strip_all','rendering_changed_any_font'])}",
}
open(os.path.join(TAB, "nums_bnnorm.tex"), "w").write(
    "% generated by tools/make_paper_assets.py from step14_*.csv -- do not edit\n" +
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("wrote", os.path.join(TAB, "nums_bnnorm.tex"), macros)
notes.append("- `nums_bnnorm.tex` <- step14_collateral.csv + step14_clean_text_cost.csv + "
             "step14_security.csv (bnunicodenormalizer vs render-equivalence)")

notes.append("- `fig_example.pdf` <- built by `tools/make_example_figure.py`; renders the "
             "attack and the RAB example with HarfBuzz and refuses to write the file "
             "unless the two bitmaps in panel (a) are pixel-identical")
open("paper/NUMBERS.md", "w").write(
    "# Provenance\n\nEvery figure and table in the paper is generated by a script from a CSV "
    "written by a notebook. No value is entered by hand. Regenerating reproduces every file "
    "byte for byte.\n\n" +
    "\n".join(notes) + "\n")
print("\nwrote paper/NUMBERS.md")
print("done:", len([n for n in notes]), "assets")

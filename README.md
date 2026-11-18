# Invisible Characters, Visible Harm — code and results

Artifact for the ICCIT 2026 submission on zero-width evasion of Bangla content
moderation. Anonymized for double-blind review.

Two invisible Unicode characters that Bangla needs for ordinary spelling —
ZWNJ (U+200C) and ZWJ (U+200D) — can be inserted into a comment so that the text
renders identically but a moderation model stops flagging it. Whether a model is
affected is decided by its tokenizer: `BertTokenizer._clean_text` deletes every
Unicode category-C character, so BERT-family models never see the attack, while
SentencePiece and byte-pair tokenizers pass it through.

## What is here

```
src/render_attack.py          attack, three defenses, corpus statistics
src/revision_run.py           revision experiments (other invisible characters,
                              second dataset, seeds, adversarial training)
notebooks/                    every experiment, in run order
tools/make_paper_assets.py    builds all paper figures and tables from results/
tools/make_revision_tables.py builds the revision tables and numbers from
                              results/revision/
tools/make_revision_notebooks.py  builds the nb14a-f and nb15 notebooks from src/
tools/rev_checks/             CPU-side revision checks (CoreText, character
                              categories, dataset overlap)
tools/make_example_figure.py  renders the Bangla figure, refuses to write it
                              unless the two lines are pixel-identical
tools/make_form_assets.py     builds the perceptibility study stimuli
tools/score_google_form.py    scores the study, writes results/human_study.csv
study/                        stimuli, answer key, anonymised responses
results/*.csv                 every number in the paper
requirements.txt
```

Notebooks, in the order they were run:

| Notebook | What it does |
|---|---|
| `nb01_gate.ipynb` | First check: does the attack survive the standard Bangla normalizer |
| `nb02_tokenizer.ipynb` | Tokenizer survey across 10 models |
| `nb03_ceiling.ipynb` | How strong the attack can get, three position rules |
| `nb06_verified.ipynb` | Main result: XLM-R x 3 seeds, full data, all defenses |
| `nb06b_second_encoder.ipynb` | MuRIL, plus tokenizer inflation under the verified attack |
| `nb06c_fp32_and_cure.ipynb` | BanglaT5 in fp32, plus the residual flag |
| `nb06d_pipeline.ipynb` | End-to-end: unmitigated evasion |
| `nb07_llm_verified.ipynb` | Qwen2.5-7B as a zero-shot moderator |
| `nb08`-`nb13` | Putting every encoder on one protocol, and the mDeBERTa fix |
| `nb14_bnnorm_baseline.ipynb` | The published Bangla normalizer as a baseline defense |

`nb04` and `nb05` are deliberately absent. They produced the first version of
these results using an attack that was **not** verified as invisible, so their
numbers were superseded by `nb06`/`nb07` and are not part of the paper.

## Revision experiments

Added after review. All of them use the protocol of `nb13` (full training set,
2 epochs, `transformers==4.46.3`, full precision) and write their CSVs to
`results/revision/<run>/`.

| Notebook | What it does |
|---|---|
| `nb14a_revision_bdshs_seed42.ipynb` | BD-SHS, five models, seed 42: attacks with CGJ (U+034F), VS16 (U+FE0F) and LRM (U+200E) besides ZWNJ; the defense generalized to every default-ignorable code point; the decision-change rule for the residual flag |
| `nb14b_revision_bdshs_seeds.ipynb` | The same five models with seeds 43 and 44 |
| `nb14c_revision_dataset2.ipynb` | Second dataset: BLP-2025 shared task, subtask 1A, seed 42 |
| `nb14e_revision_blp_seed43.ipynb`, `nb14f_revision_blp_seed44.ipynb` | BLP-2025 with seeds 43 and 44 |
| `nb14d_revision_advtrain.ipynb` | Adversarial training of XLM-RoBERTa on ZWNJ-attacked text, tested on characters it never saw |
| `nb15_revision_llms.ipynb` | Zero-shot LLM moderators: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Qwen2.5-14B |
| `nb15b_revision_gemma.ipynb` | Gemma-2-9B again with a smaller batch, after it ran out of memory in `nb15` |

`nb14_bnnorm_baseline.ipynb` belongs to the submitted version and is unrelated to
`nb14a`-`nb14f`.

To rebuild the revision tables and every number they feed:

```bash
python tools/make_revision_tables.py results/revision out/
```

The CPU-side checks in `tools/rev_checks/` need the datasets locally;
`coretext_glyphs.py` and `coretext_breakdown.py` also need macOS and PyObjC and
are run from inside that folder. Their outputs are `results/rev_coretext.csv`
and `results/rev_indic_tokenizers.csv`.

## Threat model

An insertion counts as imperceptible only if HarfBuzz shapes the word to the
same glyph identifiers and positions, in **all eight** reference fonts, with and
without it. HarfBuzz is the shaper used by Chrome, Android and Firefox.

The eight fonts are Noto Sans Bengali, Noto Serif Bengali, Hind Siliguri, Tiro
Bangla, Baloo Da 2, Anek Bangla, Galada and Mina. `render_attack.fetch_fonts()`
downloads them; they are not redistributed here.

## Reproducing the results

Data is **not** included. BD-SHS is third-party and we do not redistribute it.
Get it from the authors of that dataset; it is also mirrored on Kaggle as
`naurosromim/bdshs`. The revision notebooks that use BLP-2025 download it from
the shared task's public repository themselves.

The notebooks were run on Kaggle with two T4 GPUs and internet enabled. Each one
is self-contained: it writes `render_attack.py` to the working directory, fetches
the fonts, and saves its CSVs. Attach the dataset and run all.

To rebuild every figure and table in the paper from the CSVs already here:

```bash
pip install -r requirements.txt
python tools/make_paper_assets.py
```

It writes to `paper/fig/` and `paper/tab/`, and refuses to run if any result
file is missing. No number in the paper is typed by hand.

## Perceptibility study

Twenty-three Bangla readers judged 30 pairs of lines each. `study/form_answer_key.csv`
gives the ground truth, `study/form_responses_anonymised.csv` the responses with
registration numbers replaced by P01-P23 and timestamps removed, and
`tools/score_google_form.py` reproduces the numbers in the paper.

## A note on library versions

Results use `transformers==4.46.3`, pinned deliberately. On version 5, mDeBERTa-v3
produces a non-finite loss on its second optimizer step, and BanglaT5 loses eleven
points of evasion because a different tokenizer is selected. `requirements.txt`
pins the version.

## Note on content

BD-SHS is a hate-speech corpus. No comment text appears in this repository, in
any notebook output, or in any CSV. Everything here is counts and rates.

## License

MIT for the code. The dataset and the fonts keep their own licenses.

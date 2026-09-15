#!/usr/bin/env python3
"""Build the supervisor briefing PDF from the same CSVs the paper uses."""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, KeepTogether)

INK, MUTE, ACC = colors.HexColor("#1A1A1A"), colors.HexColor("#5A5A5A"), colors.HexColor("#8C2F24")
RULE = colors.HexColor("#D8D4CE")
S = lambda n, **k: ParagraphStyle(n, **k)
H1 = S("H1", fontName="Times-Bold", fontSize=17, leading=21, textColor=INK, spaceAfter=3)
SUB = S("SUB", fontName="Times-Italic", fontSize=10.5, leading=14, textColor=MUTE, spaceAfter=14)
H2 = S("H2", fontName="Times-Bold", fontSize=12.5, leading=15, textColor=ACC, spaceBefore=15, spaceAfter=5)
BODY = S("B", fontName="Times-Roman", fontSize=10.3, leading=14.6, textColor=INK, spaceAfter=7)
LEAD = S("L", parent=BODY, fontSize=11.6, leading=16.4, spaceAfter=9)
Q = S("Q", fontName="Times-Bold", fontSize=10.3, leading=14, textColor=INK, spaceBefore=9, spaceAfter=2)
A = S("A", parent=BODY, leftIndent=11, spaceAfter=4)
CELL = S("C", fontName="Times-Roman", fontSize=9, leading=11.8, textColor=INK)
CELLB = S("CB", parent=CELL, fontName="Times-Bold")
CAP = S("CAP", fontName="Times-Italic", fontSize=8.8, leading=11.5, textColor=MUTE, spaceBefore=3, spaceAfter=10)

R = lambda f: pd.read_csv(f"results/{f}")
L, C, E = R("step6_ladder.csv"), R("step6c_ladder.csv"), R("step7_llm_evasion.csv")
W, P, I = R("step6_wild_collateral.csv"), R("step6d_pipeline.csv"), R("step6b_inflation.csv")
xl = lambda c: L[(L.model == "xlm-roberta-base") & (L.config == c)].evasion.mean() * 100
f1 = lambda v: f"{v:.1f}"

def table(rows, widths, head=True):
    data = [[Paragraph(c, CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
          ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]
    t.setStyle(TableStyle(st))
    return t

F = []
F.append(Paragraph("What we found, and why it holds up", H1))
F.append(Paragraph("A briefing on the ICCIT 2026 submission &mdash; prepared for discussion with the supervisor", SUB))

F.append(Paragraph("The finding in one sentence", H2))
F.append(Paragraph(
    "Bangla needs two invisible characters to spell ordinary words. An attacker can sprinkle those "
    "same characters into an abusive comment so that it looks <i>exactly</i> the same on screen but a "
    "moderation model stops flagging it &mdash; and whether a given model is affected is decided not by "
    "the model, its size or its training data, but by one line inside its tokenizer.", LEAD))

F.append(Paragraph("Why this is not obvious", H2))
F.append(Paragraph(
    "Tokenizers of the BERT family run a cleaning step, <font face='Courier'>_clean_text</font>, which "
    "silently deletes every character in Unicode category C. The zero-width joiner and non-joiner are in "
    "that category, so those models never see the attack. They are protected <i>by accident</i>. "
    "SentencePiece and byte-pair tokenizers have no such step and pass the characters straight through.", BODY))
F.append(Paragraph(
    "Nobody chose this. It appears in no model card and no documentation. That is what makes it worth "
    "a paper: the security property of a Bangla moderation system is being decided by an undocumented "
    "implementation detail that its builders never considered.", BODY))

F.append(Paragraph("The evidence", H2))
F.append(table([
    ["Model", "Tokenizer", "Clean accuracy", "Evasion rate"],
    ["BanglaBERT", "WordPiece", "0.924", "0.0%"],
    ["MuRIL", "WordPiece", "0.919", "0.0%"],
    ["XLM-RoBERTa", "SentencePiece", "0.911", f1(xl('gated_inert_ALL')) + "% &plusmn; 8.3"],
    ["BanglaT5", "SentencePiece", "0.753", f1(C[C.config=='gated_inert_ALL'].evasion.iloc[0]*100) + "%"],
], [95, 105, 90, 105]))
F.append(Paragraph(
    "The last two rows are the argument. BanglaBERT and BanglaT5 were built by the same laboratory, on "
    "Bangla corpora from the same effort. One is never fooled; the other is fooled almost nine times in ten. "
    "The tokenizer is the only meaningful difference between them, which rules out &lsquo;multilingual versus "
    "monolingual&rsquo; and &lsquo;big versus small&rsquo; as explanations.", BODY))
F.append(Paragraph(
    f"A zero-shot Qwen2.5-7B moderator loses {E[E.attack=='gated_inert_ALL'].evasion_NO.iloc[0]*100:.1f}% of "
    "the comments it had correctly flagged. Token counts tell the same story: under attack, WordPiece "
    f"tokenizers inflate by {I[I.family=='WordPiece'].inflation_pct.max():.1f}%, SentencePiece by "
    f"{I[I.family=='SentencePiece'].inflation_pct.min():.0f}&ndash;{I[I.family=='SentencePiece'].inflation_pct.max():.0f}%.", BODY))

F.append(Paragraph("Why the result is trustworthy", H2))
F.append(Paragraph(
    "This is the part worth defending in conversation, because it is where most student papers are weak.", BODY))
F.append(table([
    ["What could have gone wrong", "What we did about it"],
    ["&ldquo;Invisible&rdquo; might just be an assumption",
     "Every insertion is checked with HarfBuzz &mdash; the shaper Chrome, Android and Firefox use &mdash; in "
     "eight Bangla fonts. An insertion is kept only if the glyphs and their positions are identical. "
     "Figure 1 of the paper shows a comment before and after, and the two bitmaps match to the pixel."],
    ["Our first number was inflated",
     "An early version reached 99.6%, but that attack was visibly broken in 73.3% of comments. Once "
     "invisibility was enforced the honest figure became 73.7%. We report the lower number and say why."],
    ["Evasion might just be truncation",
     "Flooding a comment makes it longer, so toxic words could be cut off by the length limit. Measured: "
     "73.7% at 512 tokens versus 73.8% at 128. Ruled out."],
    ["The attack might not be targeted at all",
     "Correct &mdash; and we say so. The same attack removes flags from <i>ordinary</i> comments at the same "
     "rate. The models do not hide hate; they stop reading Bangla and default to &lsquo;not toxic&rsquo;. "
     "We call this failing open rather than overclaiming."],
    ["Results might be one lucky run",
     "XLM-RoBERTa was run with three seeds and is reported as mean &plusmn; standard deviation. The spread "
     "is wide (64.5&ndash;80.4%) and we print the spread rather than the best run."],
], [150, 245]))

F.append(Paragraph("The defense, and its honest limits", H2))
F.append(Paragraph(
    "We propose removing a zero-width character only when removing it does not change what is drawn, plus a "
    "flag for manipulations that remain visible. On security this ties with simply deleting every zero-width "
    "character. The difference is collateral damage, measured on the 340 real comments in the corpus that "
    "already contain these characters:", BODY))
wc = W.set_index("defense").rendering_changed_any_font
F.append(table([
    ["Defense", "Real comments visibly corrupted"],
    ["Delete every zero-width character", f"{int(wc['strip_all'])} of 340"],
    ["Rule-based (keep only near hasant)", f"{int(wc['position_aware'])} of 340"],
    ["Ours (keep only what is visible)", f"<b>{int(wc['render_equivalence'])} of 340</b>"],
], [230, 165]))
F.append(Paragraph(
    "The corrupted words are not exotic. Seven of them are the RAB pattern: the abbreviation is spelled with "
    "a zero-width joiner, and deleting it turns the joiner form into the reph form &mdash; a different word "
    "on the page.", BODY))
try:
    F.append(Image("/tmp/figure1_inspect.png", width=330, height=197))
    F.append(Paragraph("Figure 1 of the paper, enlarged. Top pair: identical. Bottom pair: RAB before and "
                       "after deleting the joiner.", CAP))
except Exception:
    pass
F.append(Paragraph(
    f"End to end, comments that evade the classifier <i>and</i> are never flagged &mdash; so no human ever "
    f"sees them &mdash; come to {P.UNMITIGATED_evasion.max()*100:.1f}% across all three attacks.", BODY))
F.append(Paragraph(
    "One caveat we state in the paper: the defense reverses the attack by construction, not by discovery. "
    "Its value is in the collateral damage numbers, not in the fact that it works.", BODY))

F.append(Paragraph("What we deliberately did not claim", H2))
for s in [
    "The attack concept is not ours. Boucher et al. (IEEE S&amp;P 2022) showed imperceptible Unicode attacks "
    "on Latin script. We extend it to Bangla, where the usual defense cannot be applied, and we identify the "
    "tokenizer mechanism. We cite them in the second paragraph.",
    "One dataset, one binary task. We do not claim the numbers transfer to other corpora.",
    "BanglaT5 reaches only 0.753 accuracy in our setup. Its 88.4% is evidence about tokenizer families, not "
    "a claim about a well-trained BanglaT5.",
    "The adaptive attack we designed to break our own rule-based defense barely works &mdash; 0.23 usable "
    "positions per comment, 0.2% evasion. We report it as a negative result.",
]:
    F.append(Paragraph("&bull;&nbsp;&nbsp;" + s, BODY))

F.append(Paragraph("Questions the supervisor is likely to ask", H2))
qa = [
 ("Is this actually new, or just Boucher applied to Bangla?",
  "Two things are new. First, the tokenizer-family mechanism, which explains why two models from the same "
  "lab differ by 88 points &mdash; that is in no prior paper. Second, Bangla genuinely cannot use the "
  "Latin-script defense, because the characters are part of the orthography. We verified that on real corpus "
  "words, not invented examples."),
 ("Why only one exposed encoder trained properly?",
  "We tried mDeBERTa-v3 and mT5 as well; neither would train on a T4. The paper says so in one line. "
  "BanglaT5 is the more useful comparison anyway, because it shares a lab and a corpus with BanglaBERT."),
 ("73.7% with a standard deviation of 8.3 is a wide spread.",
  "Agreed, and we print it. Three seeds is the minimum defensible number. The claim does not depend on the "
  "exact value: the contrast is against 0.0% for WordPiece models, and that gap is not close."),
 ("Qwen only catches 40% of hate speech to begin with.",
  "True, and it is in the Limitations. It bounds what the LLM experiment shows, which is why the headline "
  "claim rests on the encoders. The LLM result matters for direction, not magnitude."),
 ("Can reviewers reproduce this?",
  "Yes. Every figure and table is generated by a script directly from result CSVs; no number is typed by "
  "hand. Code, results and the anonymised repository link are in the paper."),
]
for q, a in qa:
    F.append(KeepTogether([Paragraph(q, Q), Paragraph(a, A)]))

F.append(Paragraph("What was actually run", H2))
F.append(table([
    ["Stage", "What it established"],
    ["Tokenizer survey, 10 models", "Which families pass the characters through"],
    ["XLM-R, full data, 3 seeds", "73.7% evasion; truncation ruled out; benign control"],
    ["BanglaBERT and MuRIL", "0.0% at every budget &mdash; the immune side"],
    ["BanglaT5, full precision", "88.4% &mdash; isolates the tokenizer as the cause"],
    ["Qwen2.5-7B zero-shot", "45.5% &mdash; the same failure in an LLM moderator"],
    ["Defense and collateral", "0 of 340 real comments damaged; 0.0% unmitigated"],
], [150, 245]))
F.append(Paragraph(
    "Roughly 45,000 comments, 8 fonts, 8 tokenizers, 5 trained models, and every number reproducible from "
    "the released notebooks.", BODY))

def furn(cv, doc):
    cv.saveState()
    cv.setStrokeColor(RULE); cv.setLineWidth(0.5)
    cv.line(18*mm, 16*mm, A4[0]-18*mm, 16*mm)
    cv.setFont("Times-Roman", 8); cv.setFillColor(MUTE)
    cv.drawString(18*mm, 11*mm, "ICCIT 2026 submission — internal briefing")
    cv.drawRightString(A4[0]-18*mm, 11*mm, f"Page {doc.page}")
    cv.restoreState()

SimpleDocTemplate("BRIEFING.pdf", pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                  topMargin=17*mm, bottomMargin=22*mm,
                  title="ICCIT 2026 briefing", author="").build(F, onFirstPage=furn, onLaterPages=furn)
print("wrote BRIEFING.pdf")

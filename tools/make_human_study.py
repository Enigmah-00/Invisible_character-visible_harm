#!/usr/bin/env python3
"""Build a perceptibility study packet.

30 trials. Each shows two lines of Bangla and asks whether they look the same.
  - 20 test pairs: clean text vs the rendering-verified attack. Truth = SAME.
  - 10 control pairs: genuinely different renderings. Truth = DIFFERENT.
The controls are the point. Without them, a participant who answers "same" to
everything would look like evidence for us.

Outputs the participant packet, an answer key, and a scoring script.
"""
import os, sys, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, KeepTogether, PageBreak)

sys.path.insert(0, "src")
import render_attack as ra

SC = os.environ.get("SCRATCH", ".")
FONT = os.path.join(SC, "NotoSansBengali-Regular.ttf")
R = ra.Renderer(ra.fetch_fonts(os.path.join(SC, "fonts_fig")))
os.makedirs("study/img", exist_ok=True)

# Neutral sentences only. No corpus text in the packet.
SENT = [
 "আমাদের দেশ বাংলাদেশ", "আজকের আকাশ খুব পরিষ্কার", "সে প্রতিদিন সকালে হাঁটে",
 "বইটি টেবিলের উপর আছে", "শীতকালে কুয়াশা পড়ে", "নদীর পানি অনেক ঠান্ডা",
 "ছেলেটি স্কুলে যায়", "আমি চা খেতে ভালোবাসি", "গ্রামের পথ অনেক সুন্দর",
 "তারা কাল ঢাকা যাবে", "বৃষ্টি থেমে গেছে", "মাঠে অনেক শিশু খেলছে",
 "দোকানটি সকাল আটটায় খোলে", "পাখিগুলো গাছে বসে আছে", "রান্নাঘরে আলো জ্বলছে",
 "ট্রেন সময়মতো এসেছে", "বাগানে নতুন ফুল ফুটেছে", "চিঠিটি গতকাল পৌঁছেছে",
 "শিক্ষক ক্লাসে পড়াচ্ছেন", "সূর্য পশ্চিমে ডুবে যায়",
]
# Control words whose rendering genuinely changes when a zero-width char is removed
# Controls are full sentences at the SAME size as the test pairs, so nothing but
# the text itself marks a control. Each contains a functional ZWJ.
Z = ra.ZWJ
# Only RA takes the reph form, so only RA + ZWJ + hasant + YA changes shape when
# the ZWJ is removed. (S + ZWJ + hasant + YA renders the same either way.)
CTRL = [
    "র" + Z + "্যাব অফিসে গেছে",
    "কলেজে র" + Z + "্যাগিং বন্ধ",
    "নতুন র" + Z + "্যাম কিনেছি",
    "আজ বড় র" + Z + "্যালি হবে",
    "তার র" + Z + "্যাংক ভালো",
]

def render(text, size=72):
    f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
    im = Image.new("L", (2600, 170), 255)
    ImageDraw.Draw(im).text((14, 14), text, font=f, fill=0)
    a = np.array(im)
    c = np.where(a.min(axis=0) < 250)[0]; r = np.where(a.min(axis=1) < 250)[0]
    return a[max(0, r[0]-6):r[-1]+7, max(0, c[0]-6):c[-1]+7]

def save(arr, path):
    Image.fromarray(arr).save(path)
    return path

rng = random.Random(11)
trials = []

for i, s in enumerate(SENT):
    atk, n = R.attack(s, "ALL", mode="inert", seed=100 + i)
    a, b = render(s), render(atk)
    if a.shape != b.shape or not np.array_equal(a, b):
        sys.exit(f"refusing to build: trial {i} is not pixel-identical")
    trials.append({"kind": "test", "truth": "SAME", "n": n,
                   "top": save(a, f"study/img/t{i}_a.png"),
                   "bot": save(b, f"study/img/t{i}_b.png")})

for j, w in enumerate(CTRL):
    bad = ra.defend_strip_all(w)
    a, b = render(w), render(bad)
    if a.shape == b.shape and np.array_equal(a, b):
        sys.exit(f"refusing to build: control {j} does not actually look different")
    trials.append({"kind": "control", "truth": "DIFFERENT", "n": 0,
                   "top": save(a, f"study/img/c{j}_a.png"),
                   "bot": save(b, f"study/img/c{j}_b.png")})
for j, w in enumerate(CTRL):                      # same word, no change: truth SAME
    a = render(w)
    trials.append({"kind": "control-same", "truth": "SAME", "n": 0,
                   "top": save(a, f"study/img/s{j}_a.png"),
                   "bot": save(a, f"study/img/s{j}_b.png")})

rng.shuffle(trials)
key = [{"trial": i + 1, "kind": t["kind"], "truth": t["truth"], "inserted": t["n"]}
       for i, t in enumerate(trials)]

INK = colors.HexColor("#1A1A1A"); MUTE = colors.HexColor("#666666")
H = ParagraphStyle("H", fontName="Times-Bold", fontSize=15, leading=18, textColor=INK, spaceAfter=4)
B = ParagraphStyle("B", fontName="Times-Roman", fontSize=10, leading=13.6, textColor=INK, spaceAfter=6)
N = ParagraphStyle("N", fontName="Times-Roman", fontSize=8.5, leading=11, textColor=MUTE)
L = ParagraphStyle("L", fontName="Times-Bold", fontSize=9.5, leading=12, textColor=INK)

F = [Paragraph("Can you see a difference?", H),
     Paragraph("Thank you for helping. This takes about five minutes and collects no personal "
               "information beyond what you write below.", B),
     Paragraph("Each numbered item below shows two lines of Bangla text, one above the other. "
               "For each item, mark whether the two lines <b>look the same</b> or "
               "<b>look different</b> to you. Judge only how they appear. Some pairs really are "
               "different, so please look carefully at each one. There is no time limit, and you "
               "may zoom in or move closer if you wish.", B),
     Paragraph("Name or ID: ______________________&nbsp;&nbsp;&nbsp;&nbsp;"
               "Do you read Bangla daily?&nbsp; yes / no &nbsp;&nbsp;&nbsp;&nbsp;"
               "Date: ____________", B),
     Spacer(1, 6)]

def block(idx, t):
    W = 150 * mm
    def img(p):
        a = np.array(Image.open(p)); h, w = a.shape
        w_pt = min(W, w * 0.62); return RLImage(p, width=w_pt, height=h * (w_pt / w))
    rows = [[Paragraph(f"{idx}.", L), img(t["top"])],
            [Paragraph("", L), img(t["bot"])],
            [Paragraph("", L), Paragraph("look the SAME  &#9744;&nbsp;&nbsp;&nbsp;&nbsp;"
                                         "look DIFFERENT  &#9744;", N)]]
    tb = Table(rows, colWidths=[11 * mm, W], hAlign="LEFT")
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("LINEBELOW", (0, 2), (-1, 2), 0.25, colors.HexColor("#DDDDDD"))]))
    return KeepTogether([tb, Spacer(1, 7)])

for i, t in enumerate(trials, 1):
    F.append(block(i, t))

SimpleDocTemplate("study/PERCEPTIBILITY_STUDY.pdf", pagesize=A4,
                  leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                  bottomMargin=16 * mm, title="Perceptibility study", author="").build(F)

import csv
with open("study/answer_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["trial", "kind", "truth", "inserted"]); w.writeheader(); w.writerows(key)

tests = [k for k in key if k["kind"] == "test"]
print(f"wrote study/PERCEPTIBILITY_STUDY.pdf  ({len(trials)} trials)")
print(f"  {len(tests)} attacked pairs (truth SAME), mean {np.mean([t['inserted'] for t in tests]):.1f} chars inserted")
print(f"  {sum(k['kind']=='control' for k in key)} control pairs that genuinely differ (truth DIFFERENT)")
print(f"  {sum(k['kind']=='control-same' for k in key)} identical pairs (truth SAME)")
print("wrote study/answer_key.csv")

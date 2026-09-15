#!/usr/bin/env python3
"""Build Google Form assets for the perceptibility study.

Each trial becomes ONE stacked image (top line above bottom line) so the pair can
never be split or reordered by the form. Also emits the Apps Script that creates
the form and its response sheet, and the answer key keyed to trial number.
"""
import os, sys, csv, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")
import render_attack as ra

SC = os.environ.get("SCRATCH", ".")
FONT = os.path.join(SC, "NotoSansBengali-Regular.ttf")
LAT = "/System/Library/Fonts/Supplemental/Arial.ttf"
R = ra.Renderer(ra.fetch_fonts(os.path.join(SC, "fonts_fig")))
OUT = "study/form_img"
os.makedirs(OUT, exist_ok=True)

SENT = [
 "আমাদের দেশ বাংলাদেশ", "আজকের আকাশ খুব পরিষ্কার", "সে প্রতিদিন সকালে হাঁটে",
 "বইটি টেবিলের উপর আছে", "শীতকালে কুয়াশা পড়ে", "নদীর পানি অনেক ঠান্ডা",
 "ছেলেটি স্কুলে যায়", "আমি চা খেতে ভালোবাসি", "গ্রামের পথ অনেক সুন্দর",
 "তারা কাল ঢাকা যাবে", "বৃষ্টি থেমে গেছে", "মাঠে অনেক শিশু খেলছে",
 "দোকানটি সকাল আটটায় খোলে", "পাখিগুলো গাছে বসে আছে", "রান্নাঘরে আলো জ্বলছে",
 "ট্রেন সময়মতো এসেছে", "বাগানে নতুন ফুল ফুটেছে", "চিঠিটি গতকাল পৌঁছেছে",
 "শিক্ষক ক্লাসে পড়াচ্ছেন", "সূর্য পশ্চিমে ডুবে যায়",
]
Z = ra.ZWJ
CTRL = [
    "র" + Z + "্যাব অফিসে গেছে", "কলেজে র" + Z + "্যাগিং বন্ধ",
    "নতুন র" + Z + "্যাম কিনেছি", "আজ বড় র" + Z + "্যালি হবে",
    "তার র" + Z + "্যাংক ভালো",
]
SIZE = 78

def strip_img(text):
    f = ImageFont.truetype(FONT, SIZE, layout_engine=ImageFont.Layout.RAQM)
    im = Image.new("L", (3000, 180), 255)
    ImageDraw.Draw(im).text((16, 16), text, font=f, fill=0)
    a = np.array(im)
    c = np.where(a.min(axis=0) < 250)[0]; r = np.where(a.min(axis=1) < 250)[0]
    return a[max(0, r[0]-6):r[-1]+7, max(0, c[0]-6):c[-1]+7]

def stack(top, bot, path):
    """One image: top line, gap, bottom line. Left-aligned, identical margins."""
    pad, gap, lblw = 26, 34, 46
    w = max(top.shape[1], bot.shape[1]) + pad*2 + lblw
    h = top.shape[0] + bot.shape[0] + gap + pad*2
    im = Image.new("L", (w, h), 255)
    im.paste(Image.fromarray(top), (pad + lblw, pad))
    im.paste(Image.fromarray(bot), (pad + lblw, pad + top.shape[0] + gap))
    d = ImageDraw.Draw(im)
    lf = ImageFont.truetype(LAT, 30)
    d.text((pad, pad + top.shape[0]//2 - 16), "1", font=lf, fill=150)
    d.text((pad, pad + top.shape[0] + gap + bot.shape[0]//2 - 16), "2", font=lf, fill=150)
    im.save(path)

trials = []
for i, s in enumerate(SENT):
    atk, n = R.attack(s, "ALL", mode="inert", seed=100 + i)
    a, b = strip_img(s), strip_img(atk)
    if a.shape != b.shape or not np.array_equal(a, b):
        sys.exit(f"refusing to build: test {i} is not pixel-identical")
    trials.append({"kind": "test", "truth": "SAME", "inserted": n, "top": a, "bot": b})
for j, w in enumerate(CTRL):
    bad = ra.defend_strip_all(w)
    a, b = strip_img(w), strip_img(bad)
    if a.shape == b.shape and np.array_equal(a, b):
        sys.exit(f"refusing to build: control {j} does not actually differ")
    trials.append({"kind": "control", "truth": "DIFFERENT", "inserted": 0, "top": a, "bot": b})
for j, w in enumerate(CTRL):
    a = strip_img(w)
    trials.append({"kind": "control-same", "truth": "SAME", "inserted": 0, "top": a, "bot": a})

random.Random(11).shuffle(trials)
key = []
for i, t in enumerate(trials, 1):
    name = f"trial_{i:02d}.png"
    stack(t["top"], t["bot"], os.path.join(OUT, name))
    key.append({"trial": i, "image": name, "kind": t["kind"],
                "truth": t["truth"], "inserted": t["inserted"]})
with open("study/form_answer_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["trial", "image", "kind", "truth", "inserted"])
    w.writeheader(); w.writerows(key)

n_test = sum(k["kind"] == "test" for k in key)
print(f"wrote {len(key)} stacked images to {OUT}/")
print(f"  {n_test} attacked (truth SAME, all verified pixel-identical)")
print(f"  {sum(k['kind']=='control' for k in key)} genuinely different (truth DIFFERENT)")
print(f"  {sum(k['kind']=='control-same' for k in key)} duplicate images (truth SAME)")
print("wrote study/form_answer_key.csv")

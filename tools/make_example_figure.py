#!/usr/bin/env python3
"""Render the Bangla example figure for the paper.

Panel (a): a neutral comment before and after the attack, rendered with the same
shaper and font a reader's browser uses. The two bitmaps are compared pixel by
pixel and the figure is only written if they are identical.

Panel (b): a real word that the strip-all defense corrupts.

Output is a raster image inside a vector PDF, which is the honest way to show a
rendering claim: these are the actual pixels, not a description of them.
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
import render_attack as ra

SP = os.environ.get("SCRATCH", ".")
FONT_PATHS = ra.fetch_fonts(os.path.join(SP, "fonts_fig"))
FONT = FONT_PATHS["NotoSansBengali"]
SIZE, PAD = 96, 14

ZWJ, ZWNJ = ra.ZWJ, ra.ZWNJ
RAB_OK = "র" + ZWJ + "্যাব"          # as people write it: ra + ZWJ + hasant + ya
RAB_BAD = ra.defend_strip_all(RAB_OK)  # what strip-all turns it into
CLEAN = "আমাদের দেশ বাংলাদেশ"


def render(text):
    f = ImageFont.truetype(FONT, SIZE, layout_engine=ImageFont.Layout.RAQM)
    tmp = Image.new("L", (2400, 220), 255)
    ImageDraw.Draw(tmp).text((PAD, PAD), text, font=f, fill=0)
    a = np.array(tmp)
    cols = np.where(a.min(axis=0) < 250)[0]
    rows = np.where(a.min(axis=1) < 250)[0]
    if len(cols) == 0:
        return a
    return a[max(0, rows[0] - 4):rows[-1] + 5, max(0, cols[0] - 4):cols[-1] + 5]


def pad_to(a, shape):
    out = np.full(shape, 255, dtype=a.dtype)
    out[:a.shape[0], :a.shape[1]] = a
    return out


renderer = ra.Renderer(FONT_PATHS)
attacked, n = renderer.attack(CLEAN, "ALL", mode="inert", seed=7)

A, B = render(CLEAN), render(attacked)
shape = (max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1]))
A, B = pad_to(A, shape), pad_to(B, shape)
identical = bool((A == B).all())
print(f"inserted {n} invisible characters")
print(f"clean bitmap {A.shape}, attacked bitmap {B.shape}")
print(f"PIXEL-IDENTICAL: {identical}   max abs difference: {int(np.abs(A.astype(int)-B.astype(int)).max())}")
if not identical:
    sys.exit("refusing to write the figure: the two renderings differ")

C, D = render(RAB_OK), render(RAB_BAD)
sh2 = (max(C.shape[0], D.shape[0]), max(C.shape[1], D.shape[1]))
C, D = pad_to(C, sh2), pad_to(D, sh2)
print(f"strip-all changes the word: {not bool((C == D).all())}")

plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
fig, ax = plt.subplots(4, 1, figsize=(3.4, 2.45),
                       gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.55})
for a_, img, lab in [
        (ax[0], A, f"(a) original comment"),
        (ax[1], B, f"after inserting {n} invisible characters"),
        (ax[2], C, "(b) RAB, as people write it"),
        (ax[3], D, "the same word after strip-all")]:
    a_.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    a_.set_title(lab, fontsize=7, pad=2, loc="left")
    a_.axis("off")
fig.savefig("paper/fig/fig_example.pdf", bbox_inches="tight", pad_inches=0.02, dpi=600)
print("wrote paper/fig/fig_example.pdf")

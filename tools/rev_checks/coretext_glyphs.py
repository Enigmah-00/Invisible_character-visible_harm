"""CoreText glyph-level comparison of clean vs attacked comments (counts only, no text printed).

identical  - same glyphs at the same positions
subpixel   - same glyphs, every position within 0.5 px (anti-aliasing only)
visible    - different glyphs (e.g. a dotted circle) or a shift of more than 0.5 px
    python coretext_glyphs.py <bdshs_test.csv> <font_dir> <out_csv>
"""
import sys, os, glob
import numpy as np
import pandas as pd
import Quartz
import CoreText
from Foundation import NSURL

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import render_attack as ra

test_csv, font_dir, out_csv = sys.argv[1:4]
SIZE = 40.0


def font_from_file(p):
    return CoreText.CTFontCreateWithFontDescriptor(
        CoreText.CTFontManagerCreateFontDescriptorsFromURL(NSURL.fileURLWithPath_(p))[0], SIZE, None)


FONTS = {os.path.basename(p)[:-4]: font_from_file(p) for p in sorted(glob.glob(os.path.join(font_dir, "*.ttf")))}
for ps in ("KohinoorBangla-Regular", "BanglaMN", "BanglaSangamMN"):
    FONTS["Apple:" + ps] = CoreText.CTFontCreateWithName(ps, SIZE, None)


def glyphs(text, font):
    """(font name, glyph id, x, y) for every glyph not produced by an invisible character."""
    s = Quartz.CFAttributedStringCreate(None, text, {CoreText.kCTFontAttributeName: font})
    line = CoreText.CTLineCreateWithAttributedString(s)
    out = []
    for run in CoreText.CTLineGetGlyphRuns(line):
        n = CoreText.CTRunGetGlyphCount(run)
        rf = CoreText.CTRunGetAttributes(run)[CoreText.kCTFontAttributeName]
        fname = CoreText.CTFontCopyPostScriptName(rf)
        gl = CoreText.CTRunGetGlyphs(run, (0, n), None)
        pos = CoreText.CTRunGetPositions(run, (0, n), None)
        idx = CoreText.CTRunGetStringIndices(run, (0, n), None)
        adv = CoreText.CTRunGetAdvances(run, (0, n), None)
        for g, p, i, a in zip(gl, pos, idx, adv):
            if ra.is_default_ignorable(text[i]) and abs(a.width) < 0.01:
                continue   # the invisible character's own empty glyph
            out.append((fname, int(g), float(p.x), float(p.y)))
    return out


def compare(a, b, font):
    ga, gb = glyphs(a, font), glyphs(b, font)
    if [(f, g) for f, g, _, _ in ga] != [(f, g) for f, g, _, _ in gb]:
        return "visible"
    shift = max((max(abs(x1 - x2), abs(y1 - y2)) for (_, _, x1, y1), (_, _, x2, y2) in zip(ga, gb)), default=0.0)
    return "identical" if shift == 0 else "subpixel" if shift <= 0.5 else "visible"


d = pd.read_csv(test_csv)
d = d[d["sentence"].astype(str).str.strip().str.len() > 0]
texts = d[d["hate speech"] == 1].sample(300, random_state=0)["sentence"].astype(str).tolist()
R = ra.Renderer(ra.fetch_fonts(font_dir))
rows = []
for name, char in (("ZWNJ", ra.ZWNJ), ("CGJ", ra.CGJ), ("VS16", ra.VS16)):
    atk = [R.attack(t, "ALL", char=char, mode="inert", seed=i)[0] for i, t in enumerate(texts)]
    pairs = [(t, a) for t, a in zip(texts, atk) if t != a]
    defended = [R.defend_render_equivalence(a, invisible=ra.is_default_ignorable) for _, a in pairs]
    for fname, font in FONTS.items():
        res = [compare(t, a, font) for t, a in pairs]
        rows.append({"attack": name, "font": fname, "n": len(res),
                     "identical": res.count("identical"), "subpixel": res.count("subpixel"),
                     "visible": res.count("visible"),
                     "defended_equals_clean": int(sum(x == t for x, (t, _) in zip(defended, pairs)))})
        r = rows[-1]
        print(f"{name:5s} {fname:30s} identical {r['identical']:3d} | subpixel {r['subpixel']:3d} | visible {r['visible']:3d} / {r['n']}")
pd.DataFrame(rows).to_csv(out_csv, index=False)

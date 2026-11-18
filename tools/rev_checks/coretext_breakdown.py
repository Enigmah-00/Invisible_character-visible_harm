"""Per font: share of attacked comments whose CoreText rendering has different glyphs,
only shifted glyphs (with the size of the shift), or is identical. Counts only."""
import sys, numpy as np, pandas as pd
DATA_CSV = sys.argv[1]
sys.argv = ["x", "/dev/null", "fonts_bn", "/dev/null"]
exec(open("coretext_glyphs.py").read().split("d = pd.read_csv")[0])
DOTTED = 0x25CC
d = pd.read_csv(DATA_CSV); d = d[d["sentence"].astype(str).str.strip().str.len() > 0]
texts = d[d["hate speech"] == 1].sample(300, random_state=0)["sentence"].astype(str).tolist()
R = ra.Renderer(ra.fetch_fonts("fonts_bn"))
rows = []
for name, char in (("ZWNJ", ra.ZWNJ), ("CGJ", ra.CGJ), ("VS16", ra.VS16)):
    atk = [R.attack(t, "ALL", char=char, mode="inert", seed=i)[0] for i, t in enumerate(texts)]
    pairs = [(t, a) for t, a in zip(texts, atk) if t != a]
    for fname, font in FONTS.items():
        glyph = shift = same = dotted = 0; shifts = []
        dg = CoreText.CTFontGetGlyphsForCharacters(font, [chr(DOTTED)], None, 1)
        for t, a in pairs:
            ga, gb = glyphs(t, font), glyphs(a, font)
            ka = [(f, g) for f, g, _, _ in ga]; kb = [(f, g) for f, g, _, _ in gb]
            if ka != kb:
                glyph += 1
                dotted += len(kb) > len(ka)   # an extra glyph is the dotted circle
                continue
            m = max((max(abs(x1 - x2), abs(y1 - y2)) for (_, _, x1, y1), (_, _, x2, y2) in zip(ga, gb)), default=0.0)
            if m > 0.5: shift += 1; shifts.append(m)
            else: same += 1
        n = len(pairs)
        rows.append({"attack": name, "font": fname, "n": n, "same_or_subpixel": same, "glyph_change": glyph,
                     "extra_glyph": dotted, "shift_only": shift,
                     "median_shift_px_at_40pt": round(float(np.median(shifts)), 2) if shifts else 0.0})
        r = rows[-1]
        print(f"{name:5s} {fname:30s} same {same:3d} | glyph change {glyph:3d} (extra glyph {dotted:3d}) | shift only {shift:3d} (median {r['median_shift_px_at_40pt']} px @40pt)")
pd.DataFrame(rows).to_csv("coretext_breakdown.csv", index=False)

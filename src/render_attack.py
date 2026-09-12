"""
Rendering-verified zero-width attacks and defenses for Bangla.

Threat model
------------
An insertion counts as imperceptible only if HarfBuzz shapes the word to the
same glyph ids and glyph positions, in every reference font, with and without
it. HarfBuzz is the shaper used by Chrome, Android and Firefox.

The earlier rule-based attacks (strict / semi) are kept for comparison, but
every number in the paper should come from the render-gated functions here.
"""
import os
import re
import random
import urllib.request
import unicodedata
from functools import lru_cache

import uharfbuzz as hb

ZWNJ, ZWJ, ZWSP = "‌", "‍", "​"
HASANT = "্"
ZW = (ZWNJ, ZWJ, ZWSP)

# Open-licensed Bangla fonts, all downloadable on Kaggle.
FONT_URLS = {
    "NotoSansBengali":  "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansBengali/unhinted/ttf/NotoSansBengali-Regular.ttf",
    "NotoSerifBengali": "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSerifBengali/unhinted/ttf/NotoSerifBengali-Regular.ttf",
    "HindSiliguri":     "https://raw.githubusercontent.com/google/fonts/main/ofl/hindsiliguri/HindSiliguri-Regular.ttf",
    "TiroBangla":       "https://raw.githubusercontent.com/google/fonts/main/ofl/tirobangla/TiroBangla-Regular.ttf",
    "BalooDa2":         "https://raw.githubusercontent.com/google/fonts/main/ofl/balooda2/BalooDa2%5Bwght%5D.ttf",
    "AnekBangla":       "https://raw.githubusercontent.com/google/fonts/main/ofl/anekbangla/AnekBangla%5Bwdth,wght%5D.ttf",
    "Galada":           "https://raw.githubusercontent.com/google/fonts/main/ofl/galada/Galada-Regular.ttf",
    "Mina":             "https://raw.githubusercontent.com/google/fonts/main/ofl/mina/Mina-Regular.ttf",
}


def fetch_fonts(dest, names=None):
    os.makedirs(dest, exist_ok=True)
    paths = {}
    for name in names or FONT_URLS:
        p = os.path.join(dest, name + ".ttf")
        if not os.path.exists(p):
            urllib.request.urlretrieve(FONT_URLS[name], p)
        paths[name] = p
    return paths


def is_bangla(ch):
    """True for one Bangla code point.

    An empty string returns False, so callers may pass a missing neighbour
    (start or end of string) without guarding first."""
    return len(ch) == 1 and 0x0980 <= ord(ch) <= 0x09FF


# ------------------------------------------------------------------ slots

def inert_slots(word):
    """Between two Bangla chars, neither of them hasant (the old `semi` set)."""
    return [i for i in range(1, len(word))
            if is_bangla(word[i - 1]) and is_bangla(word[i])
            and HASANT not in (word[i - 1], word[i])]


def strict_slots(word):
    """Both neighbours Bangla base letters (the old `strict` set)."""
    return [i for i in inert_slots(word)
            if unicodedata.category(word[i - 1]) == "Lo"
            and unicodedata.category(word[i]) == "Lo"]


def hasant_slots(word):
    """Immediately before or after a hasant. Position-aware defense keeps these."""
    out = set()
    for h, c in enumerate(word):
        if c != HASANT:
            continue
        if h > 0 and is_bangla(word[h - 1]):
            out.add(h)
        if h + 1 < len(word) and is_bangla(word[h + 1]):
            out.add(h + 1)
    return sorted(out)


SLOT_MODES = {
    "inert":  inert_slots,
    "strict": strict_slots,
    "hasant": hasant_slots,
    "both":   lambda w: inert_slots(w) + hasant_slots(w),
}


def _build(word, chosen, char):
    return "".join((char if i in chosen else "") + c for i, c in enumerate(word))


def _parts(text):
    return re.split(r"(\s+)", text)


# --------------------------------------------------------------- renderer

class Renderer:
    def __init__(self, font_paths):
        self.fonts = {n: hb.Font(hb.Face(hb.Blob.from_file_path(p), 0))
                      for n, p in font_paths.items()}
        self.names = list(self.fonts)
        self._shape = lru_cache(maxsize=2_000_000)(self._shape_uncached)

    def _shape_uncached(self, name, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        buf.flags = hb.BufferFlags.REMOVE_DEFAULT_IGNORABLES
        hb.shape(self.fonts[name], buf, {})
        return (tuple(g.codepoint for g in buf.glyph_infos),
                tuple((p.x_advance, p.y_advance, p.x_offset, p.y_offset)
                      for p in buf.glyph_positions))

    def same(self, a, b, fonts=None):
        return all(self._shape(n, a) == self._shape(n, b) for n in (fonts or self.names))

    def visible_in(self, a, b):
        """Names of fonts in which a and b render differently."""
        return [n for n in self.names if self._shape(n, a) != self._shape(n, b)]

    # ------------------------------------------------------------ attacks

    def attack(self, text, budget="ALL", char=ZWNJ, seed=0, mode="inert", fonts=None):
        """Insert zero-width chars only where the word still renders identically.

        Candidates are shuffled and accepted one at a time, each checked against
        the clean word with all previously accepted insertions in place.
        Returns (attacked_text, n_inserted)."""
        parts = _parts(text)
        rng = random.Random(seed)
        slots = SLOT_MODES[mode]
        cands = [(pi, i) for pi, p in enumerate(parts)
                 if p and not p.isspace() for i in slots(p)]
        rng.shuffle(cands)
        chosen, n = {}, 0
        for pi, i in cands:
            if budget != "ALL" and n >= budget:
                break
            trial = chosen.get(pi, set()) | {i}
            if self.same(parts[pi], _build(parts[pi], trial, char), fonts):
                chosen[pi] = trial
                n += 1
        out = "".join(_build(p, chosen[pi], char) if pi in chosen else p
                      for pi, p in enumerate(parts))
        return out, n

    # ------------------------------------------------------------ defense

    def defend_render_equivalence(self, text, fonts=None):
        """Delete a zero-width char iff removing it leaves rendering unchanged in
        every reference font. Only chars whose nearest non-zero-width neighbours
        on both sides are Bangla are considered, so emoji ZWJ sequences survive."""
        parts = _parts(text)
        for pi, w in enumerate(parts):
            if not any(z in w for z in ZW):
                continue
            i = 0
            while i < len(w):
                if w[i] in ZW:
                    l = i - 1
                    while l >= 0 and w[l] in ZW:
                        l -= 1
                    r = i + 1
                    while r < len(w) and w[r] in ZW:
                        r += 1
                    if l >= 0 and r < len(w) and is_bangla(w[l]) and is_bangla(w[r]):
                        cand = w[:i] + w[i + 1:]
                        if self.same(w, cand, fonts):
                            w = cand
                            continue
                i += 1
            parts[pi] = w
        return "".join(parts)


# ------------------------------------------------ rule-based (for comparison)

def attack_ungated(text, budget="ALL", char=ZWNJ, seed=0, mode="inert"):
    parts = _parts(text)
    rng = random.Random(seed)
    slots = SLOT_MODES[mode]
    cands = [(pi, i) for pi, p in enumerate(parts) if p and not p.isspace() for i in slots(p)]
    rng.shuffle(cands)
    if budget != "ALL":
        cands = cands[:budget]
    chosen = {}
    for pi, i in cands:
        chosen.setdefault(pi, set()).add(i)
    out = "".join(_build(p, chosen[pi], char) if pi in chosen else p for pi, p in enumerate(parts))
    return out, len(cands)


def defend_strip_all(text):
    for z in ZW:
        text = text.replace(z, "")
    return text


def defend_nfc(text):
    return unicodedata.normalize("NFC", text)


def defend_position_aware(text):
    out = []
    for i, ch in enumerate(text):
        if ch in (ZWNJ, ZWJ):
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if HASANT in (prev, nxt):
                out.append(ch)
            continue
        if ch == ZWSP:
            continue
        out.append(ch)
    return "".join(out)


def zw_stats(texts):
    docs = total = functional = 0
    for t in texts:
        hit = False
        for i, ch in enumerate(t):
            if ch not in (ZWNJ, ZWJ):
                continue
            hit = True
            total += 1
            prev = t[i - 1] if i > 0 else ""
            nxt = t[i + 1] if i + 1 < len(t) else ""
            functional += HASANT in (prev, nxt)
        docs += hit
    return dict(n_docs=len(texts), docs_with_zw=docs, total_zw=total,
                functional=functional, inert=total - functional)

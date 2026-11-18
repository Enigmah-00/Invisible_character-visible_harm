"""Which invisible characters survive each tokenizer, and which render invisibly in Bangla?"""
import os, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import render_attack as ra
from normalizer import normalize
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = {
    "ZWNJ": "‌", "ZWJ": "‍", "ZWSP": "​", "WJ": "⁠", "ZWNBSP": "﻿",
    "SHY": "­", "LRM": "‎", "RLM": "‏", "ALM": "؜", "INV_TIMES": "⁢",
    "MVS": "᠎", "CGJ": "͏", "VS1": "︀", "VS16": "️", "VS17": "\U000E0100",
    "TAG_SPACE": "\U000E0020", "HANGUL_FILLER": "ㅤ", "HW_HANGUL_FILLER": "ﾠ",
    "KHMER_INHERENT": "឴",
}
SENT = [
    "আমাদের দেশ বাংলাদেশ", "আজকের আকাশ খুব পরিষ্কার", "সে প্রতিদিন সকালে হাঁটে",
    "বইটি টেবিলের উপর আছে", "শীতকালে কুয়াশা পড়ে", "নদীর পানি অনেক ঠান্ডা",
    "ছেলেটি স্কুলে যায়", "আমি চা খেতে ভালোবাসি", "গ্রামের পথ অনেক সুন্দর",
    "তারা কাল ঢাকা যাবে", "বৃষ্টি থেমে গেছে", "মাঠে অনেক শিশু খেলছে",
]
TOK = {
    "BanglaBERT": "csebuetnlp/banglabert", "MuRIL": "google/muril-base-cased",
    "XLM-R": "xlm-roberta-base", "BanglaT5": "csebuetnlp/banglat5",
    "mDeBERTa": "microsoft/mdeberta-v3-base", "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}
toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOK.items()}
R = ra.Renderer(ra.fetch_fonts(os.path.join(HERE, "fonts_bn")))

print(f"{'char':17s} {'cat':3s} {'DI':2s} {'norm':5s} {'invis':>9s} " + " ".join(f"{k:>10s}" for k in TOK))
for name, c in CHARS.items():
    cat = unicodedata.category(c)
    survives = c in normalize("ক" + c + "খ")
    # rendering gate: share of inert slots where this char is invisible in all 8 fonts
    acc = cand = 0
    attacked = []
    for i, s in enumerate(SENT):
        a, n = R.attack(s, "ALL", char=c, mode="inert", seed=i)
        acc += n
        cand += sum(len(ra.inert_slots(w)) for w in s.split())
        attacked.append(a)
    # tokenizer: does the token sequence change after the Bangla normalizer?
    row = []
    for k, tk in toks.items():
        ch = sum(tk(normalize(a))["input_ids"] != tk(normalize(s))["input_ids"] for s, a in zip(SENT, attacked))
        row.append(f"{ch:>4d}/{len(SENT):<5d}")
    print(f"{name:17s} {cat:3s} {'y' if name else ''}  {'keep' if survives else 'DROP':5s} {acc:>4d}/{cand:<4d} " + " ".join(f"{r:>10s}" for r in row))

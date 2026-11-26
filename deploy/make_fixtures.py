"""
Expected outputs for deploy/check_static.html, which runs the browser demo's
cleaning and model on the same texts and compares.

    python deploy/make_fixtures.py build/static-space/model build/static-space/fixtures.json
"""

import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from day1_arabic_sentiment import DATA_DIR  # noqa: E402
from export_onnx import onnx_predictor  # noqa: E402
from text_clean import clean_text, has_arabic_or_emoji  # noqa: E402

# Cases the test sets may not cover
EDGE = [
    "", "   ", "hello", "12345", "١٢٣", "@محمد", "https://a.b/خدمة", "ـــ", "ً", "😍",
    "#خدمةٌ ممتااااز", "هههههه حلو", "&amp; خدمة &lt;b&gt; &quot;", "\n&amp;خدمة", "خدمة &amp",
    "<b>سيء</b> جداً", "<3 حلو", "a < b > c خدمة", "شكرا @user_1 و @مستخدم2",
    "www.site.com/x?y=1 رائع", "تمام👍🏽👍", "❤️ ممتاز", "ﷲ ﻻ ﺧﺪﻣﺔ", "خدمة‏ممتازة﻿",
    "خدمة\x1cممتازة\x85تمام", "A B خدمة", "سسسس 1111 ___", "إيه ده؟!!! 😂😂😂", "١٢٣٤ريال",
    "مـــمـــتـــاز", "آلسلام", "\r\nسطر\r\nثاني",
]


def main(model_dir: str, out_path: str):
    raw = []
    for name in ("test_clean", "test_hotels_clean", "test_shein_clean", "val_clean",
                 "neutral_probe", "sentences_30"):
        raw += pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))["text"].astype(str).tolist()
    texts = EDGE + list(dict.fromkeys(raw))

    cleaning = [{"text": t, "clean": clean_text(t), "has": has_arabic_or_emoji(t)} for t in texts]

    # The model on a spread of lengths, from the probe and the tweet test set
    sample = [c["clean"] for c in cleaning if c["has"]][len(EDGE):][::40][:300]
    probs = onnx_predictor(os.path.join(model_dir, "onnx", "model_quantized.onnx"), model_dir)(sample)
    model = [{"clean": t, "probs": [round(float(p), 6) for p in row]}
             for t, row in zip(sample, probs)]

    # The tokenizer on every 5th cleaned text, without truncation
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = [{"clean": c["clean"], "ids": tokenizer(c["clean"])["input_ids"]}
              for c in cleaning[::5] if c["clean"]]

    # Whole labelled sets, to measure the browser model's accuracy
    sets = {}
    for name in ("test_clean", "sentences_30"):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
        sets[name] = [{"clean": clean_text(t), "label": l} for t, l in zip(df["text"], df["label"])]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cleaning": cleaning, "tokens": tokens, "model": model, "sets": sets},
                  f, ensure_ascii=False)
    print(f"{len(cleaning)} cleaning, {len(tokens)} tokenizer, {len(model)} model cases -> {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:])

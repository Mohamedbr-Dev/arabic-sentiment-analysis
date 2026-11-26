"""
Score models on every test set, or compare merge shares on validation data.

    python src/evaluate.py model/arabert_sentiment model/arabert_sentiment_tweets
    python src/evaluate.py --sweep model/arabert_sentiment_tweets model/arabert_sentiment_reviews

The restaurant and company reviews are used for scoring only: neither page
states a license. --sweep never reads a test set.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from day1_arabic_sentiment import DATA_DIR, fetch
from merge_models import merge
from text_clean import clean_text, has_arabic_or_emoji

LABELS = ["negative", "neutral", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HUB = "https://huggingface.co/datasets/"
# file name -> (url pinned to one revision, SHA-256)
UNSEEN_FILES = {
    "restaurants.parquet": (
        _HUB + "hadyelsahar/ar_res_reviews/resolve/"
        "d51bf2435d030e0041344f576c5e8d7154828977/data/train-00000-of-00001.parquet",
        "140cf35289bed629d676a9db236fa9955a1609794816090defd7350c8abc9e43"),
    "companies_test.parquet": (
        _HUB + "hanaearg/arabicCompaniesReviewsSentimentAnalysis/resolve/"
        "dd4dcd8b1eb6e975856e0ad6f4655af9d991a047/data/test-00000-of-00001.parquet",
        "8f10118f5091d43a2455bf04774c215de32c0761b983cbb003e495dd213363b5"),
}


def load_predictor(model_dir: str, max_len: int = 64):
    """A function from cleaned texts to an array of class probabilities."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32).to(DEVICE).eval()   # as the API runs it

    def predict(texts: list) -> np.ndarray:
        probs = np.zeros((len(texts), len(LABELS)))
        by_length = np.argsort([len(t) for t in texts])   # less padding
        for start in range(0, len(texts), 32):
            rows = by_length[start:start + 32]
            enc = tokenizer([texts[i] for i in rows], max_length=max_len,
                            truncation=True, padding=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"]).logits
            probs[rows] = logits.softmax(-1).cpu().numpy()
        return probs

    return predict


def summarize(labels, probs: np.ndarray) -> dict:
    labels    = np.array(labels)
    predicted = np.array(LABELS)[probs.argmax(1)]
    correct   = predicted == labels
    confident = probs.max(1) >= 0.9
    present   = [l for l in LABELS if l in set(labels)]
    return {
        "n":             len(labels),
        "acc":           float(correct.mean()),
        "macro_f1":      float(f1_score(labels, predicted, labels=present, average="macro")),
        "recall":        {l: float(correct[labels == l].mean()) for l in present},
        "confident":     float(confident.mean()),
        "confident_acc": float(correct[confident].mean()) if confident.any() else None,
    }


def unseen_reviews(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Clean `text`, then drop repeats, conflicting labels and texts in `seen`."""
    df = df[df["text"].astype(str).map(has_arabic_or_emoji)]
    df = df.assign(text_clean=df["text"].astype(str).map(clean_text))
    df = df[df.groupby("text_clean")["label"].transform("nunique") == 1]
    df = df.drop_duplicates("text_clean")
    return df[~df["text_clean"].isin(seen)].reset_index(drop=True)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def load_probe(split: str) -> pd.DataFrame:
    """Hand-written neutral customer sentences; `dev` was used to pick the merge share."""
    df = _csv("neutral_probe")
    df = df[df["split"] == split]
    return df.assign(text_clean=df["text"].map(clean_text))


def load_test_sets() -> dict:
    path = {}
    for name, (url, sha256) in UNSEEN_FILES.items():
        path[name] = os.path.join(DATA_DIR, "external", name)
        fetch(path[name], url, sha256)

    seen = set(_csv("train_clean")["text_clean"]) | set(_csv("val_clean")["text_clean"])
    restaurants = pd.read_parquet(path["restaurants.parquet"])
    companies   = pd.read_parquet(path["companies_test.parquet"])
    sentences   = _csv("sentences_30")
    return {
        "tweets":       _csv("test_clean"),
        "hotels":       _csv("test_hotels_clean"),
        "shein":        _csv("test_shein_clean"),
        "restaurants*": unseen_reviews(pd.DataFrame({
            "text":  restaurants["text"],
            "label": restaurants["polarity"].map({0: "negative", 1: "positive"})}), seen),
        "companies*":   unseen_reviews(
            companies.rename(columns={"sentiment": "label"})[["text", "label"]], seen),
        "probe":        load_probe("heldout"),
        "sentences":    sentences.assign(text_clean=sentences["text"].map(clean_text)),
    }


def print_scores(model_dirs: list):
    sets = load_test_sets()
    for model_dir in model_dirs:
        predict = load_predictor(model_dir)
        print(f"\n{model_dir}")
        for name, df in sets.items():
            s = summarize(df["label"], predict(df["text_clean"].tolist()))
            sure = "-" if s["confident_acc"] is None else f"{s['confident_acc']:.3f}"
            recall = " ".join(f"{l[:3]} {r:.2f}" for l, r in s["recall"].items())
            print(f"  {name:13s} n={s['n']:6d}  acc {s['acc']:.3f}  macro-F1 {s['macro_f1']:.3f}"
                  f"  >=0.9: {s['confident']:.2f} (acc {sure})  recall {recall}", flush=True)
    print("\n* never trained on")


def print_sweep(dir_a: str, dir_b: str):
    """Accuracy on validation data and on the dev half of the probe, per merge share."""
    val, probe = _csv("val_clean"), load_probe("dev")
    for share in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = os.path.join(tmp, "merged")
            merge(dir_a, dir_b, share, out)
            predict = load_predictor(out)
            v = summarize(val["label"], predict(val["text_clean"].tolist()))
            p = summarize(probe["label"], predict(probe["text_clean"].tolist()))
            del predict      # Windows cannot delete a model file that is still mapped
        print(f"share {share:.1f}  val acc {v['acc']:.3f}  macro-F1 {v['macro_f1']:.3f}"
              f"  probe neutral {p['acc']:.3f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="model folders")
    parser.add_argument("--sweep", action="store_true", help="compare merge shares of two models")
    args = parser.parse_args()
    if args.sweep and len(args.models) != 2:
        parser.error("--sweep needs exactly two models")
    print_sweep(*args.models) if args.sweep else print_scores(args.models)

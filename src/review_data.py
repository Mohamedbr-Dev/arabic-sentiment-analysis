"""
Customer reviews added to the tweets: where they come from and how their
labels map to negative / neutral / positive. Both sources state a license
(hotels: MIT, SHEIN: Apache-2.0).
"""

import pandas as pd

_HUB = "https://huggingface.co/datasets/"
_HOTELS = (_HUB + "srinivasbilla/semeval-2016-absa-reviews-arabic/resolve/"
           "d22761838261d5da4a559d50cc01fe1c00a1b3f0/hotel_review_absa_arabic_")

# file name -> (url pinned to one revision, SHA-256)
REVIEW_FILES = {
    "hotels_train.csv": (
        _HOTELS + "train.csv",
        "71c9716277a71c5cdb742e0d3a6af3d67a395ccb8b1ea68c3f1921c0a8552bf5"),
    "hotels_test.csv": (
        _HOTELS + "test.csv",
        "516c7892feda2d0f041656ba28653367da11d44edf600692619b97385a657ce6"),
    "shein.parquet": (
        _HUB + "Ruqiya/Arabic_Reviews_of_SHEIN/resolve/"
        "fb63ba1255f57054d411fe02bb5cec25cd6b150c/data/train-00000-of-00001.parquet",
        "b0b53fe0ba238395481b4f306dde3e0d3c20e8bf2b551496e48e8bbd2f35ddc6"),
}


def parse_hotels(raw: pd.DataFrame) -> pd.DataFrame:
    """The file labels each aspect; keep sentences whose aspects all agree."""
    by_sentence = raw.groupby("sentence", sort=False)["sentiment"]
    labels = by_sentence.first()[by_sentence.nunique() == 1]
    return pd.DataFrame({"text": labels.index, "label": labels.values})


def parse_shein(raw: pd.DataFrame) -> pd.DataFrame:
    """Stars to labels. Three stars are mostly mixed opinions, not neutral ones."""
    labels = raw["label"].map(
        {1: "negative", 2: "negative", 4: "positive", 5: "positive"})
    return pd.DataFrame({"text": raw["raw_text"], "label": labels}).dropna().reset_index(drop=True)

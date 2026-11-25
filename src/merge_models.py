"""
Average the weights of two models fine-tuned from the same base.

The model trained on tweets alone knows neutral text but is unsure on reviews;
the one trained on tweets + reviews is the opposite. Their average keeps most
of both, at the cost of one model at serving time.

    python src/merge_models.py model/arabert_sentiment_tweets \
        model/arabert_sentiment_reviews 0.4 model/arabert_sentiment
"""

import os
import shutil
import sys

from transformers import AutoModelForSequenceClassification


def merge(dir_a: str, dir_b: str, share_b: float, out_dir: str):
    """Write (1 - share_b) * A + share_b * B to `out_dir`, with A's tokenizer and config."""
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    model   = AutoModelForSequenceClassification.from_pretrained(dir_a)
    other   = AutoModelForSequenceClassification.from_pretrained(dir_b).state_dict()
    weights = model.state_dict()
    model.load_state_dict({
        name: _blend(w, other[name], share_b) for name, w in weights.items()
    })
    model.save_pretrained(out_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "project_config.json"):
        shutil.copy(os.path.join(dir_a, name), out_dir)


def _blend(a, b, share_b: float):
    return a + share_b * (b - a) if a.is_floating_point() else a


if __name__ == "__main__":
    a, b, share, out = sys.argv[1:]
    merge(a, b, float(share), out)

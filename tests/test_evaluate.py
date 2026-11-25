import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import evaluate
from tiny import tiny_base_model

NEG, NEU, POS = [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.02, 0.03, 0.95]


class Summarize(unittest.TestCase):
    def test_accuracy_f1_and_recall(self):
        s = evaluate.summarize(["negative", "negative", "positive", "positive"],
                               np.array([NEG, NEU, POS, POS]))
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["acc"], 0.75)
        # negative: P=1 R=1/2 -> 2/3; positive 1; neutral is absent, so not averaged in
        self.assertAlmostEqual(s["macro_f1"], (2 / 3 + 1) / 2)
        self.assertEqual(s["recall"], {"negative": 0.5, "positive": 1.0})

    def test_share_and_accuracy_of_confident_answers(self):
        s = evaluate.summarize(["positive", "positive"], np.array([POS, NEG]))
        self.assertEqual((s["confident"], s["confident_acc"]), (0.5, 1.0))

    def test_no_confident_answers(self):
        s = evaluate.summarize(["negative"], np.array([NEG]))
        self.assertEqual(s["confident"], 0.0)
        self.assertIsNone(s["confident_acc"])


class UnseenReviews(unittest.TestCase):
    def prepare(self, rows, seen=()):
        df = pd.DataFrame(rows, columns=["text", "label"])
        return evaluate.unseen_reviews(df, set(seen))["text_clean"].tolist()

    def test_text_without_arabic_is_dropped(self):
        self.assertEqual(self.prepare([("hello", "positive"), ("ممتاز", "positive")]), ["ممتاز"])

    def test_repeats_after_cleaning_count_once(self):
        self.assertEqual(self.prepare([("حلو", "positive"), ("حلووو", "positive")]), ["حلو"])

    def test_text_labelled_two_ways_is_dropped(self):
        self.assertEqual(self.prepare([("حلو", "positive"), ("حلو", "negative"),
                                       ("سيء", "negative")]), ["سيء"])

    def test_text_the_model_trained_on_is_dropped(self):
        self.assertEqual(self.prepare([("حلو", "positive"), ("سيء", "negative")], {"حلو"}), ["سيء"])


class Probabilities(unittest.TestCase):
    def test_rows_keep_their_order_whatever_the_length(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m")
            tiny_base_model(path)
            predict = evaluate.load_predictor(path, max_len=8)
            texts = ["خدمة ممتاز منتج", "سيء", "موعد منتج"]
            together = predict(texts)
            for i, text in enumerate(texts):
                np.testing.assert_allclose(together[i], predict([text])[0], atol=1e-5)


class Probe(unittest.TestCase):
    def test_halves_do_not_overlap(self):
        dev, held = evaluate.load_probe("dev"), evaluate.load_probe("heldout")
        self.assertEqual((len(dev), len(held)), (40, 40))
        self.assertFalse(set(dev["text_clean"]) & set(held["text_clean"]))
        self.assertEqual(set(dev["label"]) | set(held["label"]), {"neutral"})


if __name__ == "__main__":
    unittest.main()

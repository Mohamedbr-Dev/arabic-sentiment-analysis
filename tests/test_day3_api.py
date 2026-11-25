import importlib
import json
import os
import sys
import tempfile
import unittest

import torch
from fastapi.testclient import TestClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tiny import tiny_base_model

LABELS = ["negative", "neutral", "positive"]


class ApiCase(unittest.TestCase):
    """Serves a tiny model whose class scores are `BIAS`, or random if None."""
    BIAS = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = os.path.join(cls.tmp.name, "model")
        tiny_base_model(cls.dir, cls.BIAS)
        with open(os.path.join(cls.dir, "project_config.json"), "w", encoding="utf-8") as f:
            json.dump({"id2label": dict(enumerate(LABELS)), "max_len": 8}, f)

        os.environ["MODEL_DIR"] = cls.dir
        sys.modules.pop("src.day3_api", None)
        api = importlib.import_module("src.day3_api")
        cls.client = TestClient(api.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        del os.environ["MODEL_DIR"]
        cls.tmp.cleanup()

    def post(self, text):
        return self.client.post("/predict", json={"text": text})


class Predict(ApiCase):
    def probabilities_of(self, model_text):
        tok = AutoTokenizer.from_pretrained(self.dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(self.dir).eval()
        with torch.no_grad():
            p = torch.softmax(mdl(**tok(model_text, return_tensors="pt")).logits, -1)[0]
        return dict(zip(LABELS, p.tolist()))

    def test_raw_text_is_cleaned_the_way_training_cleans_it(self):
        got = self.post("#خدمةٌ ممتااااز").json()
        for label, p in self.probabilities_of("خدمة ممتاز").items():
            self.assertAlmostEqual(got["probabilities"][label], p, places=3)

    def test_input_is_not_padded(self):
        seen = []
        hook = sys.modules["src.day3_api"]._model.register_forward_pre_hook(
            lambda m, args, kw: seen.append(kw["input_ids"].shape[1]), with_kwargs=True)
        self.post("خدمة")
        hook.remove()
        self.assertEqual(seen, [3])   # [CLS] خدمة [SEP]

    def test_text_that_fits_is_not_marked_truncated(self):
        self.assertIs(self.post("خدمة " * 6).json()["truncated"], False)   # 6 + 2 = 8

    def test_text_that_does_not_fit_is_marked_truncated(self):
        self.assertIs(self.post("خدمة " * 7).json()["truncated"], True)


class EvenPredict(ApiCase):
    BIAS = [0.0, 0.0, 0.0]

    def test_evenly_split_scores_are_uncertain(self):
        self.assertIs(self.post("خدمة").json()["uncertain"], True)


class ConfidentPredict(ApiCase):
    BIAS = [0.0, 0.0, 5.0]

    def test_high_confidence_is_not_uncertain(self):
        got = self.post("خدمة").json()
        self.assertEqual(got["sentiment"], "positive")
        self.assertIs(got["uncertain"], False)


class Page(ApiCase):
    def test_root_is_an_arabic_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/html"))
        self.assertIn('<html lang="ar" dir="rtl">', r.text)

    def test_page_uses_the_limit_the_api_enforces(self):
        self.assertIn('data-max="2000"', self.client.get("/").text)

    def test_page_loads_nothing_from_elsewhere(self):
        self.assertNotRegex(self.client.get("/").text, r'(?:src|href)="(?:https?:)?//|@import|url\(')

    def test_health_still_answers_json(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")


class RejectedInput(ApiCase):
    def test_text_without_arabic_or_emoji_is_rejected(self):
        for text in ("", "  ", "!!!", "12345", "١٢٣", "hello", "https://a.b/خدمة",
                     "@محمد", "ـــ", "ً"):
            with self.subTest(text):
                self.assertEqual(self.post(text).status_code, 422)

    def test_arabic_mixed_with_other_content_is_accepted(self):
        for text in ("خدمة", "iPhone ممتاز 15", "شكرا @user https://a.b", "#خدمة"):
            with self.subTest(text):
                self.assertEqual(self.post(text).status_code, 200)

    def test_emoji_alone_is_accepted(self):
        self.assertEqual(self.post("😍").status_code, 200)

    def test_two_thousand_characters_are_accepted(self):
        self.assertEqual(self.post("ا" * 2000).status_code, 200)

    def test_longer_text_is_rejected_without_echoing_it(self):
        r = self.post("ا" * 2001)
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("اااا", r.text)

    def test_broken_unicode_is_rejected_not_a_server_error(self):
        r = self.client.post("/predict", content=b'{"text": "\\ud800"}',
                             headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()

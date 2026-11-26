import importlib
import json
import os
import shutil
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
    HALF = False   # serve a copy of the model saved in float16
    ENV = {}       # extra environment variables for the server

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = os.path.join(cls.tmp.name, "model")
        tiny_base_model(cls.dir, cls.BIAS)
        if cls.HALF:
            half = AutoModelForSequenceClassification.from_pretrained(cls.dir).half()
            half.save_pretrained(cls.dir + "_half")
            del half
            for name in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy(os.path.join(cls.dir, name), cls.dir + "_half")
            cls.dir += "_half"
        with open(os.path.join(cls.dir, "project_config.json"), "w", encoding="utf-8") as f:
            json.dump({"id2label": dict(enumerate(LABELS)), "max_len": 8}, f)

        cls.env = {"MODEL_DIR": cls.dir, **cls.ENV}
        os.environ.update(cls.env)
        sys.modules.pop("src.day3_api", None)
        cls.api = importlib.import_module("src.day3_api")
        cls.client = TestClient(cls.api.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        for name in cls.env:
            del os.environ[name]
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


class Limiter(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from rate_limit import RateLimiter
        self.now = 0.0
        self.limiter = RateLimiter(per_client=2, total=3, window=60, clock=lambda: self.now)

    def test_each_client_has_its_own_allowance(self):
        allow = self.limiter.allow
        self.assertEqual([allow("a"), allow("a"), allow("a")], [True, True, False])
        self.assertTrue(allow("b"))

    def test_all_clients_share_a_total(self):
        allow = self.limiter.allow
        self.assertEqual([allow("a"), allow("b"), allow("c"), allow("d")], [True, True, True, False])

    def test_allowance_returns_after_the_window(self):
        self.limiter.allow("a"), self.limiter.allow("a")
        self.now = 59.9
        self.assertFalse(self.limiter.allow("a"))
        self.now = 60.1
        self.assertTrue(self.limiter.allow("a"))

    def test_refused_requests_do_not_use_the_allowance(self):
        for _ in range(5):
            self.limiter.allow("a")
        self.now = 60.1
        self.assertTrue(self.limiter.allow("a"))

    def test_quiet_clients_are_forgotten(self):
        for client in "abc":
            self.limiter.allow(client)
        self.now = 120
        self.limiter.allow("d")
        self.assertEqual(set(self.limiter.clients), {"d"})

    def test_zero_turns_a_limit_off(self):
        limiter = type(self.limiter)(per_client=0, total=0, window=60)
        self.assertTrue(all(limiter.allow("a") for _ in range(100)))


class RateLimited(ApiCase):
    ENV = {"RATE_LIMIT_PER_CLIENT": "2", "RATE_LIMIT_TOTAL": "100"}

    def setUp(self):
        self.api._limiter.clients.clear()
        self.api._limiter.all.clear()

    def post_from(self, forwarded_for):
        return self.client.post("/predict", json={"text": "خدمة"},
                                headers={"X-Forwarded-For": forwarded_for})

    def test_too_many_requests_get_429(self):
        codes = [self.post_from("1.1.1.1").status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_client_is_the_address_the_proxy_saw(self):
        # The proxy appends the address it saw; anything before it came from the client.
        for fake in ("9.9.9.1", "9.9.9.2", "9.9.9.3"):
            last = self.post_from(f"{fake}, 1.1.1.1")
        self.assertEqual(last.status_code, 429)

    def test_page_and_health_are_not_limited(self):
        for _ in range(3):
            self.post_from("1.1.1.1")
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)


class HalfPrecisionModel(ApiCase):
    HALF = True

    def test_weights_stored_in_float16_are_run_in_float32(self):
        self.assertEqual(next(self.api._model.parameters()).dtype, torch.float32)
        self.assertEqual(self.post("خدمة").status_code, 200)


class Unlimited(ApiCase):
    def test_no_limit_unless_configured(self):
        self.assertTrue(all(self.post("خدمة").status_code == 200 for _ in range(30)))


if __name__ == "__main__":
    unittest.main()

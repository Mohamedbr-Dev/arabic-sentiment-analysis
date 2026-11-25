import os
import sys
import tempfile
import unittest

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from merge_models import merge
from tiny import tiny_base_model


class Merge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a, self.b, self.out = (os.path.join(self.tmp.name, n) for n in "abo")
        tiny_base_model(self.a, [1.0, 0.0, 0.0])
        tiny_base_model(self.b, [0.0, 0.0, 2.0])
        for d in (self.a, self.b):
            with open(os.path.join(d, "project_config.json"), "w") as f:
                f.write('{"max_len": 8}')

    def tearDown(self):
        self.tmp.cleanup()

    def test_weights_are_the_weighted_average(self):
        merge(self.a, self.b, 0.25, self.out)
        bias = AutoModelForSequenceClassification.from_pretrained(self.out).classifier.bias
        self.assertTrue(torch.allclose(bias, torch.tensor([0.75, 0.0, 0.5])))

    def test_output_is_a_complete_model_folder(self):
        merge(self.a, self.b, 0.5, self.out)
        self.assertEqual(AutoTokenizer.from_pretrained(self.out).tokenize("خدمة"), ["خدمة"])
        with open(os.path.join(self.out, "project_config.json")) as f:
            self.assertEqual(f.read(), '{"max_len": 8}')

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            merge(self.a, self.b, 0.5, self.b)


if __name__ == "__main__":
    unittest.main()

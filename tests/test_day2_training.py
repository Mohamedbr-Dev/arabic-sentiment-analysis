import os
import sys
import tempfile
import unittest

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import day2_finetune_arabert as day2
from tiny import SPECIAL, WORDS, tiny_base_model

def write_csv(path, n=6):
    names = ["negative", "neutral", "positive"]
    pd.DataFrame({"text_clean": [WORDS[i % 5] for i in range(n)],
                  "label": [names[i % 3] for i in range(n)],
                  "label_id": [i % 3 for i in range(n)]}).to_csv(path, index=False)


class Labels(unittest.TestCase):
    def test_unknown_label_is_rejected_not_turned_neutral(self):
        df = pd.DataFrame({"text_clean": ["ا", "ب"], "label": ["negative", "mixed"]})
        with self.assertRaisesRegex(ValueError, "label"):
            day2.ArabicSentimentDataset(df, None, 8)


class AlwaysNegative(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        logits = torch.tensor([[1.0, 0.0, 0.0]] * len(labels))
        return type("Out", (), {"loss": torch.tensor(0.0), "logits": logits})


class Metrics(unittest.TestCase):
    def test_f1_weighs_every_class_equally_even_absent_ones(self):
        batch = {"input_ids": torch.zeros(4, 2), "attention_mask": torch.ones(4, 2),
                 "label": torch.tensor([0, 0, 0, 1])}
        f1 = day2.evaluate_epoch(AlwaysNegative(), [batch], "cpu")[2]
        # negative: P=3/4 R=1 -> 6/7; neutral 0; positive 0
        self.assertAlmostEqual(f1, (6 / 7) / 3)


class Training(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        tiny_base_model(os.path.join(d, "base"))
        write_csv(os.path.join(d, "train.csv"))
        write_csv(os.path.join(d, "val.csv"))
        self.config = {**day2.CONFIG,
                       "model_name": os.path.join(d, "base"), "max_len": 8,
                       "batch_size": 2, "epochs": 1,
                       "train_path": os.path.join(d, "train.csv"),
                       "val_path": os.path.join(d, "val.csv"),
                       "test_path": os.path.join(d, "test.csv"),
                       "model_dir": os.path.join(d, "out")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_training_never_reads_the_test_set(self):
        # test.csv does not exist: touching it would raise
        history = day2.train_model(self.config)[2]
        self.assertEqual(len(history["val_f1"]), 1)

    def test_cleaned_text_reaches_the_model_unaltered(self):
        tok = AutoTokenizer.from_pretrained(self.config["model_name"])
        df = pd.DataFrame({"text_clean": ["خدمة"], "label": ["neutral"]})
        ids = day2.ArabicSentimentDataset(df, tok, 8)[0]["input_ids"].tolist()
        self.assertEqual(ids[1], (SPECIAL + WORDS).index("خدمة"))

    def test_saved_config_names_the_classes(self):
        day2.train_model(self.config)
        cfg = AutoConfig.from_pretrained(self.config["model_dir"])
        self.assertEqual(cfg.id2label, {0: "negative", 1: "neutral", 2: "positive"})

    def test_saved_model_is_scored_on_the_test_set(self):
        day2.train_model(self.config)
        write_csv(self.config["test_path"], n=9)
        _, _, _, preds, labels = day2.evaluate_on_test(self.config)
        self.assertEqual([int(x) for x in labels], [0, 1, 2] * 3)
        self.assertEqual(len(preds), 9)

    def test_another_test_file_can_be_scored(self):
        day2.train_model(self.config)
        other = os.path.join(self.tmp.name, "test_hotels.csv")
        write_csv(other, n=3)
        self.assertEqual(len(day2.evaluate_on_test(self.config, other)[3]), 3)


if __name__ == "__main__":
    unittest.main()

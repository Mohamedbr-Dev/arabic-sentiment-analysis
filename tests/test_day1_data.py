import json
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import day1_arabic_sentiment as day1


def arsas_rows(*rows):
    """(text, label id, confidence) rows in the published ArSAS schema."""
    return pd.DataFrame({
        "#Tweet_ID": [str(929241870508724224 + i) for i in range(len(rows))],
        "Tweet_text": [r[0] for r in rows],
        "Topic": "Event",
        "Sentiment_label_confidence": [r[2] for r in rows],
        "Speech_act_label": "Assertion",
        "Speech_act_label_confidence": "1.0",
        "label": [r[1] for r in rows],
    })


class ParseArsas(unittest.TestCase):
    def test_maps_numeric_classes_to_names(self):
        df = day1.parse_arsas(arsas_rows(
            ("نص سلبي", 0, "1.0"), ("نص محايد", 1, "1.0"), ("نص ايجابي", 2, "1.0")))
        self.assertEqual(df["label"].tolist(), ["negative", "neutral", "positive"])
        self.assertEqual(df["text"].tolist(), ["نص سلبي", "نص محايد", "نص ايجابي"])

    def test_drops_mixed_class(self):
        df = day1.parse_arsas(arsas_rows(("مختلط", 3, "1.0"), ("سلبي", 0, "1.0")))
        self.assertEqual(df["text"].tolist(), ["سلبي"])

    def test_drops_rows_holding_several_merged_tweets(self):
        merged = "تغريدة\tEvent\tNegative\t1\tExpression\t1\r\n929676697372446720\tاخرى"
        df = day1.parse_arsas(arsas_rows((merged, 0, "1.0"), ("سليمة", 0, "1.0")))
        self.assertEqual(df["text"].tolist(), ["سليمة"])

    def test_drops_rows_whose_annotators_all_disagreed(self):
        df = day1.parse_arsas(arsas_rows(("خلاف", 0, "0.35"), ("اغلبية", 0, "0.66")))
        self.assertEqual(df["text"].tolist(), ["اغلبية"])


class LoadArsas(unittest.TestCase):
    def test_corrupt_file_raises_instead_of_using_sample_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arsas.parquet")
            with open(path, "wb") as f:
                f.write(b"not the dataset")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                day1.load_arsas_dataset(path)


class Preprocess(unittest.TestCase):
    def test_text_labels_become_integer_ids(self):
        df = pd.DataFrame({"text": ["خدمة سيئة جدا", "موعد يوم الاحد", "خدمة ممتازة جدا"],
                           "label": ["negative", "neutral", "positive"]})
        out = day1.preprocess_dataframe(df)
        self.assertEqual(out["label_id"].tolist(), [0, 1, 2])
        self.assertTrue(pd.api.types.is_integer_dtype(out["label_id"]))
        self.assertEqual(out["label"].tolist(), ["negative", "neutral", "positive"])

    def test_model_text_keeps_letters_emoji_and_hashtag_words(self):
        df = pd.DataFrame({"text": ["خدمةٌ سيئة😍 #منتدى_شباب"], "label": ["negative"]})
        self.assertEqual(day1.preprocess_dataframe(df)["text_clean"].tolist(),
                         ["خدمة سيئة 😍 منتدى شباب"])

    def test_unknown_label_is_rejected(self):
        df = pd.DataFrame({"text": ["خدمة سيئة جدا"], "label": ["mixed"]})
        with self.assertRaisesRegex(ValueError, "mixed"):
            day1.preprocess_dataframe(df)

    def test_drops_text_empty_after_cleaning(self):
        df = pd.DataFrame({"text": ["http://a.b @x", "خدمة سيئة جدا"],
                           "label": ["neutral", "negative"]})
        self.assertEqual(day1.preprocess_dataframe(df)["text"].tolist(), ["خدمة سيئة جدا"])


class Deduplicate(unittest.TestCase):
    def test_keeps_one_row_per_cleaned_text(self):
        df = pd.DataFrame({"dedup_key": ["ا", "ا", "ب"], "label_id": [0, 0, 1]})
        self.assertEqual(day1.deduplicate(df)["dedup_key"].tolist(), ["ا", "ب"])

    def test_drops_text_labelled_two_ways(self):
        df = pd.DataFrame({"dedup_key": ["ا", "ا", "ب"], "label_id": [0, 2, 1]})
        self.assertEqual(day1.deduplicate(df)["dedup_key"].tolist(), ["ب"])

    def test_repost_with_different_tail_is_dropped(self):
        body = "واحد اثنان ثلاثه اربعه خمسه سته سبعه ثمانيه"
        df = pd.DataFrame({"dedup_key": [body + " عاجل", body + " المزيد هنا"],
                           "label_id": [1, 1]})
        self.assertEqual(day1.deduplicate(df)["dedup_key"].tolist(), [body + " عاجل"])

    def test_repost_with_added_first_word_is_dropped(self):
        body = "واحد اثنان ثلاثه اربعه خمسه سته سبعه ثمانيه"
        df = pd.DataFrame({"dedup_key": [body, "شاهد " + body], "label_id": [1, 1]})
        self.assertEqual(day1.deduplicate(df)["dedup_key"].tolist(), [body])

    def test_different_texts_sharing_some_words_are_kept(self):
        df = pd.DataFrame({"dedup_key": ["واحد اثنان ثلاثه", "واحد اثنان اربعه"],
                           "label_id": [1, 1]})
        self.assertEqual(len(day1.deduplicate(df)), 2)


class Split(unittest.TestCase):
    def setUp(self):
        labels = [0] * 500 + [1] * 300 + [2] * 200
        self.df = pd.DataFrame({"dedup_key": [f"نص {i}" for i in range(1000)],
                                "label_id": labels})

    def test_sizes_are_80_10_10(self):
        self.assertEqual([len(p) for p in day1.split_data(self.df)], [800, 100, 100])

    def test_parts_share_no_text_and_lose_none(self):
        parts = day1.split_data(self.df)
        texts = [t for p in parts for t in p["dedup_key"]]
        self.assertEqual(len(texts), 1000)
        self.assertEqual(len(set(texts)), 1000)

    def test_each_part_keeps_class_proportions(self):
        for part, n in zip(day1.split_data(self.df), (800, 100, 100)):
            counts = part["label_id"].value_counts()
            self.assertEqual([counts[0], counts[1], counts[2]],
                             [n * 0.5, n * 0.3, n * 0.2])

    def test_same_split_every_run(self):
        a, b = day1.split_data(self.df)[2], day1.split_data(self.df)[2]
        self.assertEqual(a["dedup_key"].tolist(), b["dedup_key"].tolist())


class SampleData(unittest.TestCase):
    def test_has_no_duplicate_sentences(self):
        self.assertEqual(len(day1.generate_sample_data()), 30)
        df = day1.deduplicate(day1.preprocess_dataframe(day1.generate_sample_data()))
        self.assertEqual(len(df), 30)


class Save(unittest.TestCase):
    def test_writes_three_splits_readable_with_labels(self):
        df = pd.DataFrame({"text": ["خدمة سيئة جدا"], "label": ["negative"]})
        part = day1.preprocess_dataframe(df)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "new")
            day1.save_clean_data(part, part, part, out)
            for name in ("train", "val", "test"):
                back = pd.read_csv(os.path.join(out, f"{name}_clean.csv"))
                self.assertEqual(back["label"].tolist(), ["negative"])
                self.assertEqual(back["label_id"].tolist(), [0])
                self.assertNotIn("dedup_key", back.columns)
            with open(os.path.join(out, "dataset_stats.json"), encoding="utf-8") as f:
                stats = json.load(f)
            self.assertEqual(stats["val"], {"size": 1, "labels": {"negative": 1}})


if __name__ == "__main__":
    unittest.main()

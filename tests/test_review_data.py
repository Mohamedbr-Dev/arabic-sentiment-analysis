import os
import sys
import tempfile
import unittest

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import day1_arabic_sentiment as day1
from review_data import parse_hotels, parse_shein


def hotels(*rows):
    return pd.DataFrame([
        {"sentence": s, "aspect": "x", "sentiment": l, "category": "HOTEL#GENERAL"}
        for s, l in rows])


class ParseHotels(unittest.TestCase):
    def test_one_row_per_sentence_when_aspects_agree(self):
        df = parse_hotels(hotels(("غرفة نظيفة", "positive"), ("غرفة نظيفة", "positive")))
        self.assertEqual(df.values.tolist(), [["غرفة نظيفة", "positive"]])

    def test_drops_sentence_whose_aspects_disagree(self):
        df = parse_hotels(hotels(("موقع جيد وخدمة سيئة", "positive"),
                                 ("موقع جيد وخدمة سيئة", "negative"),
                                 ("غرفة قذرة", "negative")))
        self.assertEqual(df.values.tolist(), [["غرفة قذرة", "negative"]])


class ParseShein(unittest.TestCase):
    def test_stars_become_labels_and_three_stars_are_dropped(self):
        raw = pd.DataFrame({"raw_text": list("ابتثج"), "text": list("12345"),
                            "label": [1, 2, 3, 4, 5]})
        self.assertEqual(parse_shein(raw).values.tolist(), [
            ["ا", "negative"], ["ب", "negative"], ["ث", "positive"], ["ج", "positive"]])


class PublishedTestSet(unittest.TestCase):
    """A source that ships its own test set keeps it, minus nothing to training."""

    def frame(self, n, prefix):
        return pd.DataFrame({
            "text": [f"{prefix} {chr(0x0628 + i % 20) * 3}{chr(0x0628 + i // 20) * 4}" for i in range(n)],
            "label": [["negative", "positive"][i % 2] for i in range(n)]})

    def test_training_rows_that_repeat_a_test_row_are_dropped(self):
        test = self.frame(10, "اختبار")
        train = pd.concat([self.frame(100, "تدريب"), test.head(4)])
        tr, va, te = day1.split_with_published_test(train, test)
        self.assertEqual(sorted(te["text"]), sorted(test["text"]))
        self.assertEqual(len(tr) + len(va), 100)
        self.assertFalse(set(te["text"]) & (set(tr["text"]) | set(va["text"])))

    def test_a_tenth_of_training_becomes_validation(self):
        tr, va, _ = day1.split_with_published_test(
            self.frame(100, "تدريب"), self.frame(10, "اختبار"))
        self.assertEqual((len(tr), len(va)), (90, 10))


class ShortReviews(unittest.TestCase):
    def test_short_text_is_kept_when_asked(self):
        df = pd.DataFrame({"text": ["حلو"], "label": ["positive"]})
        self.assertEqual(len(day1.preprocess_dataframe(df, min_chars=1)), 1)


class Elongation(unittest.TestCase):
    def prepared(self, labels):
        df = pd.DataFrame({"text": ["حلو", "حلووووو", "سيء جدا"], "label": labels})
        return day1.deduplicate(day1.preprocess_dataframe(df, min_chars=1),
                                keys=day1.REVIEW_KEYS)["text"].tolist()

    def test_texts_equal_after_cleaning_count_once(self):
        self.assertEqual(self.prepared(["positive", "positive", "negative"]), ["حلو", "سيء جدا"])

    def test_texts_equal_after_cleaning_with_two_labels_are_dropped(self):
        self.assertEqual(self.prepared(["positive", "negative", "negative"]), ["سيء جدا"])


class SplitSizes(unittest.TestCase):
    def test_custom_fractions(self):
        df = pd.DataFrame({"text_clean": [str(i) for i in range(100)], "label_id": [0, 1] * 50})
        self.assertEqual([len(p) for p in day1.split_data(df, val=0.1, test=0.3)], [60, 10, 30])


class SaveExtraTests(unittest.TestCase):
    def test_each_extra_test_set_gets_its_own_file(self):
        part = day1.preprocess_dataframe(
            pd.DataFrame({"text": ["خدمة سيئة جدا"], "label": ["negative"]}))
        with tempfile.TemporaryDirectory() as d:
            day1.save_clean_data(part, part, part, d, extra_tests={"hotels": part})
            back = pd.read_csv(os.path.join(d, "test_hotels_clean.csv"))
            self.assertEqual(back["label"].tolist(), ["negative"])


class Fetch(unittest.TestCase):
    def test_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f")
            with open(path, "wb") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                day1.fetch(path, "https://unused.invalid", "0" * 64)


if __name__ == "__main__":
    unittest.main()

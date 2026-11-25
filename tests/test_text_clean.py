import os
import sys
import unittest

sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from text_clean import clean_text


class CleanText(unittest.TestCase):
    CASES = [
        # name, input, expected
        ("diacritics removed",       "الخدمة سيئة\u064c جدا\u064b",      "الخدمة سيئة جدا"),
        ("letter forms kept",        "رائعة إلى أحمد مستشفى",  "رائعة إلى أحمد مستشفى"),
        ("tatweel removed",          "جمـــيل",                "جميل"),
        ("elongation collapsed",     "رااااائع",               "رائع"),
        ("laughter kept short",      "ههههههه",                "هه"),
        ("double letters kept",      "الله ممتاز",             "الله ممتاز"),
        ("number repeats kept",      "سنة 2000",               "سنة 2000"),
        ("link becomes a word",      "خبر https://t.co/abc",   "خبر رابط"),
        ("bare www link",            "خبر www.a.com",          "خبر رابط"),
        ("mention becomes a word",   "شكرا @user_1",           "شكرا مستخدم"),
        ("hashtag words kept",       "#منتدى_شباب_العالم",     "منتدى شباب العالم"),
        ("emoji separated",          "رائع😍😍جدا",            "رائع 😍 😍 جدا"),
        ("emoji variant normalised", "شكرا❤\ufe0f",            "شكرا ❤"),
        ("skin tone dropped",        "👍\U0001F3FD",           "👍"),
        ("direction marks dropped",  "نص\u200f عربي\u200e",    "نص عربي"),
        ("ligature expanded",        "ل\ufef7سف",              "للأسف"),
        ("html tag removed",         "نص &lt;img src='a.png'/&gt; عربي", "نص عربي"),
        ("html entity decoded",      "قال &amp; كتب",          "قال & كتب"),
        ("comparison signs kept",    "3 < 5 > 2",              "3 < 5 > 2"),
        ("whitespace collapsed",    "  نص \n\t عربي ",        "نص عربي"),
    ]

    def test_cases(self):
        for name, raw, want in self.CASES:
            with self.subTest(name):
                self.assertEqual(clean_text(raw), want)

    def test_cleaning_twice_changes_nothing(self):
        for name, raw, _ in self.CASES:
            with self.subTest(name):
                once = clean_text(raw)
                self.assertEqual(clean_text(once), once)

    def test_non_string_is_empty(self):
        self.assertEqual(clean_text(None), "")


if __name__ == "__main__":
    unittest.main()

"""
Text cleaning shared by data preparation, training and the API.

The model must see the same text in all three, so this is the only place
that prepares input for it. Rules follow what the AraBERT-twitter tokenizer
knows: diacritics and emoji glued to a word become [UNK], plain letter
forms (ة ى ئ أ) are in its vocabulary and stay as written.
"""

import html
import re
import unicodedata

_INVISIBLE  = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\ufe0e\ufe0f]")
_SKIN_TONE  = re.compile("[\U0001F3FB-\U0001F3FF]")
_DIACRITICS = re.compile("[\u064b-\u065f\u0670ـ]")      # harakat + tatweel
_TAG        = re.compile(r"</?[a-zA-Z][^<>]*>")
_URL        = re.compile(r"(?:https?://|www\.)\S+")
_MENTION    = re.compile(r"@\w+")
_LAUGH      = re.compile("ه{3,}")
_ELONGATED  = re.compile(r"([^\W\d_])\1{2,}")
_EMOJI      = re.compile("([\U0001F000-\U0001FAFF☀-➿])")

_ARABIC     = re.compile("[\u0621-\u063a\u0641-\u064a]")


def has_arabic_or_emoji(text: str) -> bool:
    """True if anything besides links and mentions carries Arabic or emoji."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = _MENTION.sub(" ", _URL.sub(" ", _TAG.sub(" ", text)))
    return bool(_ARABIC.search(text) or _EMOJI.search(text))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = _TAG.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = _SKIN_TONE.sub("", text)
    text = _URL.sub(" رابط ", text)
    text = _MENTION.sub(" مستخدم ", text)
    text = re.sub("[#_]", " ", text)
    text = _DIACRITICS.sub("", text)
    text = _LAUGH.sub("هه", text)
    text = _ELONGATED.sub(r"\1", text)
    text = _EMOJI.sub(r" \1 ", text)
    return " ".join(text.split())

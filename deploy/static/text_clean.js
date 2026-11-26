// The browser twin of src/text_clean.py: the model must see the text it was trained on.
// deploy/check_static.html compares the two on every test text.

const INVISIBLE  = /[​-‏‪-‮⁦-⁩﻿︎️]/gu;
const SKIN_TONE  = /[\u{1F3FB}-\u{1F3FF}]/gu;
const DIACRITICS = /[ً-ٰٟـ]/gu;
const TAG        = /<\/?[a-zA-Z][^<>]*>/gu;
const URL_       = /(?:https?:\/\/|www\.)[^\s\x1c-\x1f\x85]+/gu;
const MENTION    = /@[\p{L}\p{N}_]+/gu;                    // Python's \w
const LAUGH      = /ه{3,}/gu;
const ELONGATED  = /([\p{L}\p{Nl}\p{No}])\1{2,}/gu;        // Python's [^\W\d_]
const EMOJI      = /([\u{1F000}-\u{1FAFF}☀-➿])/gu;
const ARABIC     = /[ء-غف-ي]/u;
const SPACES     = /[\s\x1c-\x1f\x85]+/gu;                 // Python's str.split()

// html.unescape: a textarea reads entities but not tags.
const decoder = typeof document === "undefined" ? null : document.createElement("textarea");
function unescape(text) {
  if (!decoder || !text.includes("&")) return text;
  decoder.innerHTML = text;
  return decoder.value;
}

function prepare(text) {
  return unescape(text).normalize("NFKC");
}

export function hasArabicOrEmoji(text) {
  text = prepare(text).replace(TAG, " ").replace(URL_, " ").replace(MENTION, " ");
  return ARABIC.test(text) || new RegExp(EMOJI.source, "u").test(text);
}

export function cleanText(text) {
  if (typeof text !== "string") return "";
  return prepare(text)
    .replace(TAG, " ")
    .replace(INVISIBLE, "")
    .replace(SKIN_TONE, "")
    .replace(URL_, " رابط ")
    .replace(MENTION, " مستخدم ")
    .replace(/[#_]/gu, " ")
    .replace(DIACRITICS, "")
    .replace(LAUGH, "هه")
    .replace(ELONGATED, "$1")
    .replace(EMOJI, " $1 ")
    .split(SPACES).filter(Boolean).join(" ");
}

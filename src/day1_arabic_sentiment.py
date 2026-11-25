"""
=============================================================
  Arabic Sentiment Analysis — اليوم الأول
  تحميل البيانات + التنظيف + الاستكشاف
=============================================================
Stack: Python + Pandas + scikit-learn + Matplotlib
Dataset: ArSAS (Arabic Sentiment Analysis)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from review_data import REVIEW_FILES, parse_hotels, parse_shein
from text_clean import clean_text

_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(_ROOT, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw", "arsas.parquet")

# Published copy of ArSAS, pinned to one revision so the data cannot change silently.
ARSAS_URL = (
    "https://huggingface.co/datasets/arbml/ArSAS/resolve/"
    "619c18ba46019c28099c82a430e773e98471b5db/"
    "data/train-00000-of-00001-34595f80773a1fab.parquet"
)
ARSAS_SHA256 = "8160f85fd50c9893dd9fae1c459b017ebc13456c50d5b47d75791b4e688f2558"
ARSAS_LABELS = {0: "negative", 1: "neutral", 2: "positive"}   # 3 = Mixed, unused

# Below 0.5 the three annotators each chose a different label.
MIN_CONFIDENCE = 0.5

# TF-IDF cosine above this = the same text reposted with small changes.
NEAR_DUPLICATE = 0.7

# Short reviews repeat as "حلو" / "حلووو": dedup_key keeps them apart, text_clean does not.
REVIEW_KEYS = ("text_clean", "dedup_key")

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
SEED = 42

# ─────────────────────────────────────────
# 1. تحميل الـ Dataset
# ─────────────────────────────────────────

def fetch(path: str, url: str, sha256: str):
    """تنزيل الملف إن لم يوجد، والتحقق من بصمته في كل مرة."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            content = f.read()
    else:
        print(f"⬇️  تنزيل {os.path.basename(path)}")
        with urllib.request.urlopen(url, timeout=60) as r:
            content = r.read()

    actual = hashlib.sha256(content).hexdigest()
    if actual != sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)


def load_arsas_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    """تحميل ArSAS من القرص، أو تنزيله أول مرة. يرفع خطأ عند أي فشل."""
    fetch(path, ARSAS_URL, ARSAS_SHA256)
    return parse_arsas(pd.read_parquet(path))


def load_reviews(folder: str = os.path.join(DATA_DIR, "external")) -> dict:
    """مراجعات العملاء: {المصدر: (train, val, test)}."""
    path = {name: os.path.join(folder, name) for name in REVIEW_FILES}
    for name, (url, sha256) in REVIEW_FILES.items():
        fetch(path[name], url, sha256)

    shein = parse_shein(pd.read_parquet(path["shein.parquet"]))
    return {
        "hotels": split_with_published_test(
            parse_hotels(pd.read_csv(path["hotels_train.csv"])),
            parse_hotels(pd.read_csv(path["hotels_test.csv"]))),
        "shein": split_data(
            deduplicate(preprocess_dataframe(shein, min_chars=1), REVIEW_KEYS), val=0.1, test=0.3),
    }


def parse_arsas(raw: pd.DataFrame) -> pd.DataFrame:
    """تحويل جدول ArSAS المنشور إلى عمودين: text و label."""
    # الملف المنشور يدمج أحياناً عدة تغريدات في خلية واحدة (TSV لم يُحلَّل)
    merged = raw["Tweet_text"].str.contains("\t", regex=False)
    agreed = raw["Sentiment_label_confidence"].astype(float) >= MIN_CONFIDENCE
    known  = raw["label"].isin(list(ARSAS_LABELS))
    print(f"   مستبعد: Mixed={(~known).sum()} | مدموج={merged.sum()} | خلاف={(~agreed).sum()}")

    df = raw[known & agreed & ~merged]
    return pd.DataFrame({
        "text":  df["Tweet_text"].str.strip(),
        "label": df["label"].map(ARSAS_LABELS),
    }).reset_index(drop=True)


def generate_sample_data() -> pd.DataFrame:
    """30 جملة للتطوير فقط (--sample). لا تصلح لتقييم النموذج."""
    samples = [
        # Positive (إيجابي)
        ("الخدمة ممتازة وسريعة جداً، شكراً جزيلاً", "positive"),
        ("المنتج رائع وجودته عالية، سأشتريه مرة أخرى", "positive"),
        ("تجربة رائعة لم أتوقعها، أنصح الجميع بها", "positive"),
        ("أحببت كثيراً هذا المكان، الجو جميل والناس طيبون", "positive"),
        ("الفيلم ممتع ومشوق، قضيت وقتاً رائعاً", "positive"),
        ("الطعام لذيذ جداً والكميات كافية", "positive"),
        ("موظفو الخدمة محترمون ومتعاونون جداً", "positive"),
        ("المشروع نجح بامتياز بفضل الفريق المتميز", "positive"),
        ("سعيد جداً بهذا الاختيار ولن أندم عليه", "positive"),
        ("التطبيق سهل الاستخدام ويعمل بشكل مثالي", "positive"),

        # Negative (سلبي)
        ("الخدمة سيئة جداً ولم أجد من يساعدني", "negative"),
        ("المنتج رديء الجودة وتوقف عن العمل بعد يوم واحد", "negative"),
        ("تجربة مزعجة ومؤلمة لن أكررها أبداً", "negative"),
        ("المكان قذر والخدمة بطيئة وغير منظمة", "negative"),
        ("الفيلم ممل ومكرر ولم يستحق المشاهدة", "negative"),
        ("الطعام بارد وطعمه سيء ولن أعود مجدداً", "negative"),
        ("الموظفون غير مهتمين بالعملاء ومتعجرفون", "negative"),
        ("المشروع فشل بسبب سوء التخطيط والإدارة", "negative"),
        ("نادم على هذا الاختيار وضياع المال والوقت", "negative"),
        ("التطبيق يتعطل باستمرار ومليء بالأخطاء", "negative"),

        # Neutral (محايد)
        ("المنتج وصل في الموعد المحدد", "neutral"),
        ("تم الرد على استفساري خلال ساعتين", "neutral"),
        ("الفيلم عبارة عن قصة عائلية اعتيادية", "neutral"),
        ("يمكن الحجز عبر الموقع الرسمي", "neutral"),
        ("المطعم يفتح من الساعة 12 ظهراً حتى 11 مساءً", "neutral"),
        ("التوصيل متاح لجميع مناطق المدينة", "neutral"),
        ("تضمن الجلسة مناقشة عدة نقاط رئيسية", "neutral"),
        ("الشركة أعلنت عن إطلاق منتج جديد الشهر القادم", "neutral"),
        ("الاجتماع سيعقد يوم الثلاثاء القادم", "neutral"),
        ("النسخة الجديدة تحتوي على بعض التحديثات", "neutral"),
    ]

    return pd.DataFrame(samples, columns=["text", "label"])


# ─────────────────────────────────────────
# 2. تنظيف النصوص العربية
# ─────────────────────────────────────────

def dedup_key(text: str) -> str:
    """
    صيغة مختزلة لكشف التكرار فقط، ولا تصل إلى النموذج (انظر text_clean.py):
    - إزالة الروابط والإيموجي والرموز الخاصة
    - توحيد الهمزات والألف
    - إزالة التشكيل (diacritics)
    - تنظيف المسافات الزائدة
    """
    if not isinstance(text, str):
        return ""

    # إزالة الروابط
    text = re.sub(r'http\S+|www\S+', '', text)

    # إزالة المنشن والهاشتاق
    text = re.sub(r'@\w+|#\w+', '', text)

    # إزالة الإيموجي
    text = re.sub(
        r'[\U00010000-\U0010ffff'
        r'\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF]+',
        '', text, flags=re.UNICODE
    )

    # إزالة التشكيل (harakat)
    text = re.sub(r'[\u064B-\u065F]', '', text)

    # توحيد الهمزات
    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'[ىئ]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)

    # إزالة الأرقام والرموز غير العربية
    text = re.sub(r'ـ+', '', text)  # إزالة التطويل (الكشيدة)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

    # تنظيف المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame, min_chars: int = 6) -> pd.DataFrame:
    """تطبيق التنظيف على الـ DataFrame كاملاً"""
    df = df.copy()

    # تنظيف النص
    df['text_clean'] = df['text'].apply(clean_text)
    df['dedup_key']  = df['text'].apply(dedup_key)

    # إزالة النصوص الفارغة بعد التنظيف
    df = df[df['dedup_key'].str.len() >= min_chars].reset_index(drop=True)

    # إضافة إحصائيات مفيدة
    df['word_count']  = df['text_clean'].apply(lambda x: len(x.split()))
    df['char_count']  = df['text_clean'].apply(len)

    # ترميز الـ labels
    df['label']    = df['label'].str.lower().str.strip()
    df['label_id'] = df['label'].map(LABEL_MAP)
    if df['label_id'].isna().any():
        unknown = df.loc[df['label_id'].isna(), 'label'].unique().tolist()
        raise ValueError(f"Unknown labels: {unknown}")
    df['label_id'] = df['label_id'].astype(int)

    return df


def deduplicate(df: pd.DataFrame, keys=("dedup_key",)) -> pd.DataFrame:
    """
    حذف النصوص ذات التسميات المتعارضة، ثم إبقاء نسخة واحدة من كل نص.
    يشمل ذلك إعادة النشر بكلمات زائدة، ولو تُركت لتسربت من التدريب إلى الاختبار.
    """
    for key in keys:
        conflicting = df.groupby(key)["label_id"].transform("nunique") > 1
        df = df[~conflicting].drop_duplicates(key)
    df = df.reset_index(drop=True)

    vectors = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+",
    ).fit_transform(df["dedup_key"])
    kept = np.ones(len(df), dtype=bool)
    for start in range(0, len(df), 1000):          # chunked to bound memory
        similar = (vectors[start:start + 1000] @ vectors.T).tocoo()
        pairs = sorted(
            (start + row, col)
            for row, col, score in zip(similar.row, similar.col, similar.data)
            if score > NEAR_DUPLICATE and col < start + row
        )
        for later, earlier in pairs:               # drop only if the earlier one stays
            if kept[earlier]:
                kept[later] = False
    return df[kept].reset_index(drop=True)


def split_data(df: pd.DataFrame, val: float = 0.1, test: float = 0.1, seed: int = SEED):
    """تقسيم train/val/test (80/10/10 افتراضياً) مع الحفاظ على نسب الفئات."""
    df_train, rest = train_test_split(
        df, test_size=val + test, stratify=df["label_id"], random_state=seed)
    df_val, df_test = train_test_split(
        rest, test_size=test / (val + test), stratify=rest["label_id"], random_state=seed)
    return tuple(p.reset_index(drop=True) for p in (df_train, df_val, df_test))


def split_with_published_test(train: pd.DataFrame, test: pd.DataFrame, seed: int = SEED):
    """الاحتفاظ باختبار المصدر، وحذف ما يكرره من التدريب، وعُشر التدريب للتحقق."""
    both = preprocess_dataframe(pd.concat([
        test.assign(is_test=True), train.assign(is_test=False)]), min_chars=1)
    both = deduplicate(both, REVIEW_KEYS)     # test rows come first, so they are the copy that stays
    is_test = both.pop("is_test")
    df_train, df_val = train_test_split(
        both[~is_test], test_size=0.1, stratify=both.loc[~is_test, "label_id"], random_state=seed)
    return tuple(p.reset_index(drop=True) for p in (df_train, df_val, both[is_test]))


# ─────────────────────────────────────────
# 3. استكشاف البيانات (EDA)
# ─────────────────────────────────────────

def explore_data(df: pd.DataFrame, title: str = "Dataset"):
    """طباعة إحصائيات أساسية"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  عدد الأمثلة    : {len(df):,}")
    print(f"  الأعمدة         : {list(df.columns)}")
    print(f"\n  توزيع التسميات:")
    dist = df['label'].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct / 3)
        print(f"    {label:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  إحصائيات طول النص:")
    print(f"    متوسط الكلمات : {df['word_count'].mean():.1f}")
    print(f"    أقصر نص       : {df['word_count'].min()} كلمات")
    print(f"    أطول نص        : {df['word_count'].max()} كلمات")
    print(f"\n  مثال إيجابي:")
    pos = df[df['label'] == 'positive']
    if len(pos):
        print(f"    {pos['text'].iloc[0]}")
    print(f"\n  مثال سلبي:")
    neg = df[df['label'] == 'negative']
    if len(neg):
        print(f"    {neg['text'].iloc[0]}")


def plot_eda(df_train: pd.DataFrame, df_test: pd.DataFrame, output_path: str):
    """رسم مخططات EDA احترافية"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Arabic Sentiment Analysis — EDA", fontsize=14, fontweight='bold')

    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}

    # 1. توزيع التسميات في Train
    train_dist = df_train['label'].value_counts()
    axes[0].bar(
        train_dist.index,
        train_dist.values,
        color=[colors.get(l, '#999') for l in train_dist.index]
    )
    axes[0].set_title('Label Distribution (Train)')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    for i, (label, val) in enumerate(train_dist.items()):
        axes[0].text(i, val + 0.5, str(val), ha='center', fontweight='bold')

    # 2. توزيع طول النصوص
    for label in df_train['label'].unique():
        subset = df_train[df_train['label'] == label]['word_count']
        axes[1].hist(subset, alpha=0.6, label=label, color=colors.get(label, '#999'), bins=15)
    axes[1].set_title('Word Count Distribution')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # 3. مقارنة Train vs Test
    train_pct = df_train['label'].value_counts(normalize=True) * 100
    test_pct  = df_test['label'].value_counts(normalize=True) * 100
    x = np.arange(len(train_pct))
    w = 0.35
    axes[2].bar(x - w/2, train_pct.values, w, label='Train', color='#3498db', alpha=0.8)
    axes[2].bar(x + w/2, test_pct.reindex(train_pct.index).fillna(0).values,
                w, label='Test', color='#e67e22', alpha=0.8)
    axes[2].set_title('Train vs Test Distribution (%)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(train_pct.index)
    axes[2].set_ylabel('Percentage (%)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ تم حفظ المخططات في: {output_path}")
    plt.close()


# ─────────────────────────────────────────
# 4. حفظ البيانات المنظفة
# ─────────────────────────────────────────

def save_clean_data(df_train: pd.DataFrame, df_val: pd.DataFrame,
                    df_test: pd.DataFrame, out_dir: str = DATA_DIR, extra_tests: dict = {}):
    """حفظ البيانات المنظفة للمرحلة القادمة"""
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    parts = {"train": df_train, "val": df_val, "test": df_test}
    parts.update({f"test_{name}": df for name, df in extra_tests.items()})
    for name, df in parts.items():
        df.drop(columns="dedup_key").to_csv(os.path.join(out_dir, f"{name}_clean.csv"), index=False, encoding='utf-8-sig')
        stats[name] = {"size": len(df), "labels": df['label'].value_counts().to_dict()}

    with open(os.path.join(out_dir, "dataset_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✅ تم الحفظ في {out_dir}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(use_sample: bool):
    if use_sample:
        raw, out_dir = generate_sample_data(), os.path.join(DATA_DIR, "sample")
    else:
        raw, out_dir = load_arsas_dataset(), DATA_DIR
    print(f"✅ محمّل: {len(raw):,}")

    clean  = preprocess_dataframe(raw)
    unique = deduplicate(clean)
    print(f"🧹 بعد التنظيف: {len(clean):,} | بعد حذف التكرار والتعارض: {len(unique):,}")

    df_train, df_val, df_test = split_data(unique)
    extra_tests = {}
    if not use_sample:
        for name, (train, val, test) in load_reviews().items():
            print(f"➕ {name}: train={len(train):,} val={len(val):,} test={len(test):,}")
            df_train = pd.concat([df_train, train], ignore_index=True)
            df_val   = pd.concat([df_val, val], ignore_index=True)
            extra_tests[name] = test

    explore_data(df_train, "Train Set")
    explore_data(df_val,   "Validation Set")
    explore_data(df_test,  "Test Set")

    save_clean_data(df_train, df_val, df_test, out_dir, extra_tests)
    plot_eda(df_train, df_test, os.path.join(out_dir, "eda_plots.png"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
    parser = argparse.ArgumentParser(description="تحميل البيانات وتنظيفها وتقسيمها")
    parser.add_argument("--sample", action="store_true",
                        help="30 جملة تجريبية إلى data/sample بدل ArSAS")
    main(parser.parse_args().sample)

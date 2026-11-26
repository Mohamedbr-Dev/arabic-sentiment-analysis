# Arabic Sentiment Analysis | تحليل المشاعر العربية

Fine-tuned **AraBERT** (aubmindlab/bert-base-arabertv02-twitter) for 3-class Arabic sentiment classification (positive / neutral / negative), served via **FastAPI**.

**Live demo:** https://medtn12-arabic-sentiment-analysis.static.hf.space (the model runs in your browser; first load downloads about 130 MB).

This repository holds the code, the tests and the two hand-written test files. The datasets and the trained models (518 MB each) are not included: step 1 below downloads the data, and step 2 trains the model.

---

## Project Structure

```
arabic-sentiment-analysis/
├── data/
│   ├── raw/arsas.parquet      # ArSAS as published (downloaded on first run)
│   ├── external/              # hotel and SHEIN reviews (downloaded on first run)
│   ├── train_clean.csv        # 12,406 tweets + 3,628 hotel + 1,081 SHEIN reviews
│   ├── val_clean.csv          # 1,551 + 404 + 180 - picks the best epoch
│   ├── test_clean.csv         # 1,551 tweets - scored once, at the end
│   ├── test_hotels_clean.csv  # 1,057 reviews
│   ├── test_shein_clean.csv   # 542 reviews
│   ├── neutral_probe.csv      # 80 hand-written neutral sentences (40 dev, 40 held out)
│   ├── sentences_30.csv       # 30 hand-written sentences, 10 per class
│   ├── dataset_stats.json
│   └── old_synthetic/         # the previous 48/12-row files, unused
├── src/
│   ├── day1_arabic_sentiment.py   # data loading, cleaning, EDA
│   ├── day2_finetune_arabert.py   # AraBERT fine-tuning
│   ├── review_data.py             # review sources and their labels
│   ├── evaluate.py                # scores models on every test set
│   ├── merge_models.py            # averages two trained models
│   ├── text_clean.py              # cleaning shared by data, training and API
│   ├── static/index.html          # page to try the model
│   └── day3_api.py                # FastAPI: the page and the API
├── tests/                         # python -m unittest discover -s tests
├── model/
│   ├── arabert_sentiment/         # served model (merged)
│   ├── arabert_sentiment_tweets/  # trained on tweets only
│   ├── arabert_sentiment_reviews/ # trained on tweets + reviews
│   └── arabert_sentiment_old/     # the original model, unused
├── requirements.txt
└── README.md
```

## Label Mapping

| Label    | ID |
|----------|----|
| negative | 0  |
| neutral  | 1  |
| positive | 2  |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Day 1 — Data loading & EDA

```bash
python src/day1_arabic_sentiment.py
```

Downloads ArSAS ([arbml/ArSAS](https://huggingface.co/datasets/arbml/ArSAS), checksum-verified; the page states no license), drops the Mixed class, malformed rows and labels all annotators disagreed on, removes duplicates and near-duplicates, then makes a stratified 80/10/10 split. Adds hotel reviews ([SemEval-2016](https://huggingface.co/datasets/srinivasbilla/semeval-2016-absa-reviews-arabic), MIT, own test set kept) and [SHEIN reviews](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN) (Apache-2.0, 60/10/30) to train and validation.

Outputs in `data/`: `train_clean.csv`, `val_clean.csv`, `test_clean.csv`, `dataset_stats.json`, `eda_plots.png`. Use `--sample` for 30 toy sentences in `data/sample/`.

### 3. Day 2 — Fine-tune AraBERT

```bash
python src/day2_finetune_arabert.py
```

Downloads `aubmindlab/bert-base-arabertv02-twitter` from HuggingFace and fine-tunes for 3 epochs. Best checkpoint (highest macro-F1 on the validation split) is saved to `model/arabert_sentiment/`.

Hyperparameters:

| Param      | Value |
|------------|-------|
| MAX_LEN    | 64    |
| BATCH_SIZE | 16    |
| EPOCHS     | 3     |
| LR         | 2e-5  |

The served model is not a direct output of this step: it averages the weights of two such runs, 60% trained on tweets alone and 40% on tweets + reviews. Retraining overwrites `model/arabert_sentiment/`, so move the served model first. To rebuild it:

```bash
python src/merge_models.py model/arabert_sentiment_tweets model/arabert_sentiment_reviews 0.4 model/arabert_sentiment
```

To reproduce the table below (about 25,000 texts per model), or the validation scores behind the 0.4 share:

```bash
python src/evaluate.py model/arabert_sentiment model/arabert_sentiment_reviews model/arabert_sentiment_tweets
python src/evaluate.py --sweep model/arabert_sentiment_tweets model/arabert_sentiment_reviews
```

The restaurant and company reviews are downloaded for scoring only; their pages state no license, so do not redistribute them.

Results (accuracy / macro-F1 over the classes present):

| Test set | n | Tweets only | Tweets + reviews | Served (merged) |
|---|---|---|---|---|
| Tweets | 1,551 | 0.871 / 0.863 | 0.875 / 0.866 | 0.871 / 0.862 |
| Hotels | 1,057 | 0.740 / 0.594 | 0.923 / 0.696 | 0.892 / 0.668 |
| SHEIN (no neutral) | 542 | 0.771 / 0.732 | 0.867 / 0.793 | 0.849 / 0.771 |
| Restaurants, never trained on (no neutral) | 8,235 | 0.729 / 0.766 | 0.853 / 0.833 | 0.827 / 0.817 |
| Companies, never trained on | 13,096 | 0.570 / 0.547 | 0.600 / 0.499 | 0.596 / 0.523 |
| Neutral customer sentences, hand-written | 40 | 0.975 | 0.350 | 0.850 |

The merge share was chosen on validation data. It was accepted although hotels lost 3.1 points against a limit of 3 set beforehand.

Known weakness: mixed opinions ("the food is good but the service is slow") get a positive or negative label. Neutral recall is 0.18 on hotels and 0.08 on companies, where "neutral" mostly means mixed. `training_history.*` describes the tweets + reviews run; `confusion_matrix.png` the served model on tweets.

### 4. Day 3 — Serve the API

```bash
uvicorn src.day3_api:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for a page to try the model (`src/static/index.html`, no build step). It replaces the JSON message `/` used to return; `/health` still answers JSON.

#### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "الخدمة ممتازة وسريعة جداً"}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.9465,
  "uncertain": false,
  "truncated": false,
  "probabilities": {
    "negative": 0.0242,
    "neutral":  0.0293,
    "positive": 0.9465
  }
}
```

- `uncertain`: confidence is below 0.6. On the validation set such answers were right 58% of the time.
- `truncated`: the text was longer than 64 tokens, and the model read only the start.
- Status 422: over 2000 characters, or no Arabic letters or emoji besides links and mentions.
- Status 429: too many requests. Off by default; set `RATE_LIMIT_PER_CLIENT` and `RATE_LIMIT_TOTAL` (requests per minute) when the server is public. Behind a proxy the client is the last `X-Forwarded-For` entry.

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Live demo in the browser

The public demo is a static page: [transformers.js](https://github.com/huggingface/transformers.js) runs an 8-bit ONNX copy of the model in the visitor's browser, so there is no server to pay for and texts never leave the device.

```bash
pip install onnx onnxruntime
python deploy/export_onnx.py model/arabert_sentiment build/static-space/model   # export, then score
python deploy/make_fixtures.py build/static-space/model build/static-space/fixtures.json
cp deploy/static/* build/static-space/
python -m http.server 8001 --directory build/static-space   # open /check.html
```

`deploy/static/text_clean.js` is the browser twin of `src/text_clean.py`; `check.html` compares the two cleanings, the two tokenizers and both models on the test texts. Last check: cleaning 5,427/5,427 and tokens 1,085/1,085 identical, 135/135 sample labels identical, tweet accuracy 0.868 in the browser against 0.871 for the PyTorch model. Per-channel weights matter: the default 8-bit export fell to 0.822 on validation data, per-channel kept 0.881 (0.884 unquantized).

Upload `README.md`, `index.html`, `model.js`, `text_clean.js` and `model/` to a static Hugging Face Space. Keep `fixtures.json` out: it holds dataset texts. `deploy/Dockerfile` and `deploy/build_space.py` serve the FastAPI version instead, on hosts that run containers (Hugging Face now charges for Docker Spaces).

---

## Notes

- `src/text_clean.py` prepares text for data, training and the API alike: links and mentions become words, emoji are separated, diacritics and elongation removed; letter forms stay as written. No Java/farasapy required.
- The model is saved only when validation macro-F1 improves, so the final checkpoint is the best seen across all epochs.
- GPU is used automatically if available (`torch.cuda.is_available()`). The local `.venv` has the CUDA build: run `.venv/Scripts/python src/day2_finetune_arabert.py`.
- Set `MODEL_DIR` to serve a model from another folder.

## License

The code is under the [MIT License](LICENSE). It does not cover the datasets or the base model, which keep their own terms; several of them state no license.

---

# تحليل المشاعر العربية

نموذج AraBERT مُضبَّط دقيقاً لتصنيف المشاعر العربية إلى ثلاث فئات (إيجابي / محايد / سلبي)، مُقدَّم عبر FastAPI.

## خطوات التشغيل

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. اليوم الأول: تحميل البيانات وتنظيفها
python src/day1_arabic_sentiment.py

# 3. اليوم الثاني: تدريب النموذج
python src/day2_finetune_arabert.py

# 4. اليوم الثالث: تشغيل الـ API
uvicorn src.day3_api:app --reload
```

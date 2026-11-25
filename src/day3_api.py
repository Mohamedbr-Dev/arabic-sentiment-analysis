"""
Arabic Sentiment Analysis — Day 3
FastAPI inference endpoint.

Run:
    uvicorn src.day3_api:app --reload

GET  /          a page to try the model
POST /predict
    Body   : {"text": "النص العربي هنا"}
    Returns: {"sentiment": "positive", "confidence": 0.94, "uncertain": false,
              "truncated": false, "probabilities": {...}}
"""

import json
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text_clean import clean_text, has_arabic_or_emoji

MAX_CHARS = 2000

# Below this the model was right only 58% of the time on the validation set.
MIN_CONFIDENCE = 0.6

# ─────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────

_BASE      = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(_BASE, "..", "model", "arabert_sentiment")
_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_dir: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"No trained model found at {model_dir}.\n"
            "Run 'python src/day2_finetune_arabert.py' first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_DEVICE)
    model.eval()

    with open(os.path.join(model_dir, "project_config.json"), encoding="utf-8") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    max_len  = int(meta["max_len"])

    return tokenizer, model, id2label, max_len


_tokenizer, _model, _id2label, _max_len = _load_model(_MODEL_DIR)


# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    description="تحليل المشاعر للنصوص العربية باستخدام AraBERT — aubmindlab/bert-base-arabertv02-twitter",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError):
    """Like FastAPI's own, minus the input: it can be huge or not valid Unicode."""
    return JSONResponse(status_code=422, content={"detail": [
        {"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in exc.errors()
    ]})


class PredictRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    uncertain: bool = Field(description=f"confidence is below {MIN_CONFIDENCE}")
    truncated: bool = Field(description="the model read only the start of the text")
    probabilities: dict[str, float]


with open(os.path.join(_BASE, "static", "index.html"), encoding="utf-8") as f:
    _PAGE = f.read().replace("{{MAX_CHARS}}", str(MAX_CHARS))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page():
    return _PAGE


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not has_arabic_or_emoji(request.text):
        raise HTTPException(status_code=422, detail="text must contain Arabic or emoji")

    enc = _tokenizer(
        clean_text(request.text),
        max_length=_max_len,
        truncation=True,
        return_overflowing_tokens=True,   # one extra row per part that did not fit
    )
    input_ids      = torch.tensor(enc["input_ids"][:1], device=_DEVICE)
    attention_mask = torch.tensor(enc["attention_mask"][:1], device=_DEVICE)

    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs     = F.softmax(logits, dim=-1).squeeze(0).cpu()
    pred_id   = int(probs.argmax())
    sentiment = _id2label[pred_id]
    confidence = float(probs[pred_id])

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        uncertain=confidence < MIN_CONFIDENCE,
        truncated=len(enc["input_ids"]) > 1,
        probabilities={
            _id2label[i]: round(float(probs[i]), 4)
            for i in range(len(_id2label))
        },
    )

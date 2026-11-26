"""
Export the model for the in-browser demo (transformers.js), then score it.

    python deploy/export_onnx.py model/arabert_sentiment build/static-space/model

Writes config.json, the tokenizer and onnx/model_quantized.onnx (8-bit weights),
then prints its scores beside the PyTorch model's on every test set.
"""

import json
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate  # noqa: E402

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


class Traceable(torch.nn.Module):
    """The classifier with the attention mask built here: the tracer cannot follow
    the mask code of transformers 5. Checked against the original before export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert = self.model.bert
        hidden = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        blocked = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        hidden = bert.encoder(hidden, attention_mask=blocked)[0]
        return self.model.classifier(bert.pooler(hidden))


def export(model_dir: str, out_dir: str) -> str:
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(os.path.join(out_dir, "onnx"))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    sample = AutoTokenizer.from_pretrained(model_dir)(
        ["خدمة ممتازة جدا والله", "سيء"], padding=True, return_tensors="pt")
    args = tuple(sample[name] for name in INPUTS)
    traceable = Traceable(model).eval()
    with torch.no_grad():
        torch.testing.assert_close(traceable(*args), model(*args).logits, atol=1e-4, rtol=1e-4)

    full = os.path.join(out_dir, "onnx", "model.onnx")
    axes = {name: {0: "batch", 1: "tokens"} for name in INPUTS}
    torch.onnx.export(traceable, args, full, dynamo=False,
                      input_names=INPUTS, output_names=["logits"], opset_version=17,
                      dynamic_axes={**axes, "logits": {0: "batch"}})
    small = os.path.join(out_dir, "onnx", "model_quantized.onnx")
    # Per-channel unsigned 8-bit: validation accuracy 0.881 against 0.884 unquantized;
    # the default (signed, per-tensor) fell to 0.822.
    quantize_dynamic(full, small, weight_type=QuantType.QUInt8, per_channel=True)
    os.remove(full)

    with open(os.path.join(model_dir, "project_config.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model.config.id2label = {int(k): v for k, v in meta["id2label"].items()}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.config.save_pretrained(out_dir)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(model_dir, name), out_dir)
    return small


def onnx_predictor(onnx_path: str, model_dir: str, max_len: int = 64):
    """Same contract as evaluate.load_predictor: cleaned texts -> class probabilities."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def predict(texts: list) -> np.ndarray:
        probs = np.zeros((len(texts), len(evaluate.LABELS)))
        for start in range(0, len(texts), 64):
            enc = tokenizer(texts[start:start + 64], max_length=max_len, truncation=True,
                            padding=True, return_tensors="np")
            logits = session.run(None, {name: enc[name].astype(np.int64) for name in INPUTS})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs[start:start + 64] = e / e.sum(1, keepdims=True)
        return probs

    return predict


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    model_dir, out_dir = sys.argv[1:]
    path = export(model_dir, out_dir)
    print(f"{path}: {os.path.getsize(path) / 2**20:.0f} MB")

    sets = evaluate.load_test_sets()
    for name, predict in (("pytorch", evaluate.load_predictor(model_dir)),
                          ("onnx 8-bit", onnx_predictor(path, out_dir))):
        print(f"\n{name}")
        for set_name, df in sets.items():
            s = evaluate.summarize(df["label"], predict(df["text_clean"].tolist()))
            print(f"  {set_name:13s} acc {s['acc']:.3f}  macro-F1 {s['macro_f1']:.3f}", flush=True)

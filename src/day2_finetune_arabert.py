"""
=============================================================
  Arabic Sentiment Analysis
  Fine-tuning AraBERT for Sentiment Analysis
=============================================================
Model : aubmindlab/bert-base-arabertv02-twitter
Task  : Text Classification (3 classes)
Stack : HuggingFace Transformers + PyTorch
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from text_clean import clean_text

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "..", "data")

CONFIG = {
    "model_name"   : "aubmindlab/bert-base-arabertv02-twitter",
    "revision"     : "0decc2e761667f0aea7aa94546ada5a32a6e5bb8",
    "max_len"      : 64,   # the base model was pre-trained on 64 tokens
    "batch_size"   : 16,   # set to 8 if CPU is slow
    "epochs"       : 3,
    "lr"           : 2e-5,
    "warmup_ratio" : 0.1,
    "seed"         : 42,
    "patience"     : 2,    # early-stopping patience (epochs)
    "train_path"   : os.path.join(_DATA_DIR, "train_clean.csv"),
    "val_path"     : os.path.join(_DATA_DIR, "val_clean.csv"),
    "test_path"    : os.path.join(_DATA_DIR, "test_clean.csv"),
    "extra_tests"  : {     # other domains, scored separately after training
        "hotels": os.path.join(_DATA_DIR, "test_hotels_clean.csv"),
        "shein" : os.path.join(_DATA_DIR, "test_shein_clean.csv"),
    },
    "model_dir"    : os.path.join(_SRC_DIR, "..", "model", "arabert_sentiment"),
    "output_dir"   : os.path.join(_SRC_DIR, ".."),
    "labels"       : ["negative", "neutral", "positive"],   # id: 0, 1, 2
}

LABEL_MAP = {l: i for i, l in enumerate(CONFIG["labels"])}   # str -> int
ID2LABEL  = {i: l for i, l in enumerate(CONFIG["labels"])}   # int -> str

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────

def _resolve_label_col(df: pd.DataFrame) -> pd.Series:
    """
    Return integer labels regardless of which column has valid data
    and regardless of pandas version (2.x object vs 3.x StringDtype).
    """
    for col_name in ("label", "label_id"):
        if col_name not in df.columns:
            continue
        col = df[col_name]
        # Always try string mapping first — works for both dtypes
        try:
            mapped = col.astype(str).str.lower().str.strip().map(LABEL_MAP)
            if mapped.notna().any():
                return mapped
        except Exception:
            pass
        # Numeric fallback
        if pd.api.types.is_numeric_dtype(col) and col.notna().any():
            return col
    raise ValueError("No usable label column (expected 'label' or 'label_id').")


class ArabicSentimentDataset(Dataset):
    """
    PyTorch Dataset for Arabic Sentiment.
    Accepts a DataFrame and returns tensors ready for the model.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df["text_clean"].fillna("").astype(str).tolist()
        labels         = _resolve_label_col(df)
        if labels.isna().any():
            raise ValueError(f"{labels.isna().sum()} rows have no usable label")
        self.labels    = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]   # cleaned by day 1
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────
# 2. Training / Evaluation loops
# ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch_num, total_epochs):
    """Train one epoch — returns average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += outputs.loss.item()
        pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device):
    """Evaluate on val/test — returns (avg_loss, acc, f1, preds, labels)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  [eval]", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, labels=list(ID2LABEL),
                        average="macro", zero_division=0)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, f1, all_preds, all_labels


# ─────────────────────────────────────────
# 3. Main training function
# ─────────────────────────────────────────

def train_model(config: dict):
    set_seed(config["seed"])

    # ── Load data ──
    print("\nLoading data...")
    df_train = pd.read_csv(config["train_path"])
    df_val   = pd.read_csv(config["val_path"])
    print(f"   Train: {len(df_train)} | Val: {len(df_val)}")

    label_dist = _resolve_label_col(df_train).map(ID2LABEL).value_counts()
    print("   Label distribution (train):\n" + label_dist.to_string())

    # ── Tokenizer & Model ──
    print(f"\nLoading {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        revision   = config["revision"],
        num_labels = len(config["labels"]),
        id2label   = ID2LABEL,
        label2id   = LABEL_MAP,
        ignore_mismatched_sizes = True,
    )
    model.to(DEVICE)

    # ── DataLoaders ──
    train_ds = ArabicSentimentDataset(df_train, tokenizer, config["max_len"])
    val_ds   = ArabicSentimentDataset(df_val,   tokenizer, config["max_len"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)

    # ── Optimizer & Scheduler ──
    optimizer    = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Training loop with early stopping ──
    print(f"\nTraining {config['epochs']} epochs on {DEVICE}...")
    history     = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1     = 0.0
    best_preds  = None
    best_labels = None
    no_improve  = 0   # early-stopping counter

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n{'─'*45}")
        print(f"  Epoch {epoch}/{config['epochs']}")
        print(f"{'─'*45}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            DEVICE, epoch, config["epochs"]
        )
        val_loss, val_acc, val_f1, preds, labels = evaluate_epoch(model, val_loader, DEVICE)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss,   6))
        history["val_acc"].append(round(val_acc,    6))
        history["val_f1"].append(round(val_f1,     6))

        print(f"\n  Epoch {epoch} results:")
        print(f"     Train Loss : {train_loss:.4f}")
        print(f"     Val Loss   : {val_loss:.4f}")
        print(f"     Val Acc    : {val_acc:.4f}  ({val_acc*100:.1f}%)")
        print(f"     Val F1     : {val_f1:.4f}")

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1     = val_f1
            best_preds  = preds
            best_labels = labels
            no_improve  = 0
            save_model(model, tokenizer, config)
            print(f"     [NEW BEST] F1={best_f1:.4f} — model saved")
        else:
            no_improve += 1
            print(f"     [no improvement] patience {no_improve}/{config['patience']}")
            if no_improve >= config["patience"]:
                print(f"\n  Early stopping triggered after epoch {epoch}.")
                break

    return model, tokenizer, history, best_preds, best_labels, config["labels"]


def evaluate_on_test(config: dict, path: str = None):
    """Score the saved best checkpoint on the held-out test set, or on another file."""
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_dir"]).to(DEVICE)
    test_ds = ArabicSentimentDataset(
        pd.read_csv(path or config["test_path"]), tokenizer, config["max_len"])
    return evaluate_epoch(model, DataLoader(test_ds, batch_size=config["batch_size"]), DEVICE)


# ─────────────────────────────────────────
# 4. Save model
# ─────────────────────────────────────────

def save_model(model, tokenizer, config: dict):
    os.makedirs(config["model_dir"], exist_ok=True)
    model.save_pretrained(config["model_dir"])
    tokenizer.save_pretrained(config["model_dir"])

    project_config = {
        "model_name" : config["model_name"],
        "labels"     : config["labels"],
        "label2id"   : {l: i for i, l in enumerate(config["labels"])},
        "id2label"   : {i: l for i, l in enumerate(config["labels"])},
        "max_len"    : config["max_len"],
    }
    with open(os.path.join(config["model_dir"], "project_config.json"), "w", encoding="utf-8") as f:
        json.dump(project_config, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────

def plot_training_history(history: dict, output_dir: str = "."):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("AraBERT Fine-tuning — Training History", fontsize=13, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], "g-o", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_f1"], "m-o", label="Val F1")
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1")
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_confusion_matrix(labels, preds, label_names: list, output_dir: str = "."):
    all_ids = list(range(len(label_names)))
    cm = confusion_matrix(labels, preds, labels=all_ids)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title("Confusion Matrix — Best Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_classification_report(labels, preds, label_names: list):
    all_ids = list(range(len(label_names)))
    print("\n" + "="*50)
    print("  Classification Report — Best Model")
    print("="*50)
    print(classification_report(labels, preds,
                                labels=all_ids,
                                target_names=label_names,
                                zero_division=0))


# ─────────────────────────────────────────
# 6. Inference
# ─────────────────────────────────────────

def predict_sentiment(text: str, model_dir: str = None):
    """
    Inference — accepts Arabic text, returns sentiment dict.
    Used later by FastAPI.
    """
    if model_dir is None:
        model_dir = CONFIG["model_dir"]

    cfg_path = os.path.join(model_dir, "project_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No project_config.json in {model_dir}. Train first.")

    with open(cfg_path, encoding="utf-8") as f:
        proj_cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    mdl       = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()

    clean = clean_text(text)
    enc   = tokenizer(
        clean,
        max_length     = proj_cfg["max_len"],
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    with torch.no_grad():
        probs = torch.softmax(
            mdl(**enc).logits, dim=1
        )[0]

    pred_id   = int(torch.argmax(probs))
    id2label  = {int(k): v for k, v in proj_cfg["id2label"].items()}

    return {
        "sentiment"    : id2label[pred_id],
        "confidence"   : round(float(probs[pred_id]), 4),
        "probabilities": {id2label[i]: round(float(p), 4)
                          for i, p in enumerate(probs)},
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles may default to cp1252
    print("Arabic Sentiment Analysis — Fine-tuning AraBERT\n")

    # ── Train ──
    model, tokenizer, history, _, _, label_names = train_model(CONFIG)

    # ── Final score: best checkpoint, test set, once ──
    _, test_acc, test_f1, test_preds, test_labels = evaluate_on_test(CONFIG)

    # ── Save history JSON ──
    hist_path = os.path.join(CONFIG["output_dir"], "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_training_history(history, CONFIG["output_dir"])
    plot_confusion_matrix(test_labels, test_preds, label_names, CONFIG["output_dir"])
    print_classification_report(test_labels, test_preds, label_names)

    # ── Summary ──
    print("\n" + "="*45)
    print("  TRAINING SUMMARY")
    print("="*45)
    print(f"  Best Val F1  : {max(history['val_f1']):.4f}")
    print(f"  Best Val Acc : {max(history['val_acc']):.4f}  ({max(history['val_acc'])*100:.1f}%)")
    print(f"  Test F1      : {test_f1:.4f}")
    print(f"  Test Acc     : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    for name, path in CONFIG["extra_tests"].items():
        # no macro-F1: it would count a class the set does not contain as 0
        acc = evaluate_on_test(CONFIG, path)[1]
        print(f"  Test {name:8s}: acc {acc:.4f}")
    print(f"  Epochs run   : {len(history['train_loss'])}")
    print(f"  Model saved  : {CONFIG['model_dir']}")
    print(f"  History JSON : {hist_path}")
    print("="*45)

    if test_f1 >= 0.75:
        print("  Target F1 >= 0.75 achieved!")
    else:
        print("  Below target F1=0.75.")

    # ── Inference demo ──
    print("\nInference test:")
    test_texts = [
        "الخدمة ممتازة وسريعة جداً، شكراً جزيلاً",
        "المنتج رديء الجودة وتوقف عن العمل",
        "التوصيل متاح لجميع مناطق المدينة",
    ]
    sentiment_label = {"positive": "(+)", "negative": "(-)", "neutral": "( )"}
    for txt in test_texts:
        result = predict_sentiment(txt)
        tag = sentiment_label.get(result["sentiment"], "")
        print(f"\n  Text : {txt}")
        print(f"  =>    {tag} {result['sentiment'].upper()}  ({result['confidence']*100:.1f}%)")
        print(f"        neg={result['probabilities']['negative']:.3f}  "
              f"neu={result['probabilities']['neutral']:.3f}  "
              f"pos={result['probabilities']['positive']:.3f}")

    print("\nDone! Next: uvicorn src.day3_api:app --reload")

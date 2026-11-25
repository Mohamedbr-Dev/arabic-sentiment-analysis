"""A 1-layer random BERT, so training and serving run in milliseconds."""
import os

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

WORDS = ["ممتاز", "سيء", "موعد", "خدمة", "منتج"]
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def tiny_base_model(path, bias=None):
    """`bias`: fixed class scores for every input, instead of random ones."""
    os.makedirs(path)
    vocab = os.path.join(path, "vocab.txt")
    with open(vocab, "w", encoding="utf-8") as f:
        f.write("\n".join(SPECIAL + WORDS))
    BertTokenizer(vocab).save_pretrained(path)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=1,
        intermediate_size=8, max_position_embeddings=16, num_labels=3,
    ))
    if bias is not None:
        torch.nn.init.zeros_(model.classifier.weight)
        model.classifier.bias.data = torch.tensor(bias)
    model.save_pretrained(path)

"""
Assemble the files of the Hugging Face Space that serves the live demo.

    python deploy/build_space.py model/arabert_sentiment build/space

The model is stored in float16 (half the download) and run in float32.
"""

import os
import shutil
import sys

from transformers import AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_FILES = ("tokenizer.json", "tokenizer_config.json", "project_config.json")
APP_FILES = ("__init__.py", "day3_api.py", "rate_limit.py", "text_clean.py", "static/index.html")


def build(model_dir: str, out_dir: str):
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    for name in ("Dockerfile", "README.md", "requirements.txt"):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(os.path.join(HERE, name), out_dir)
    shutil.copy(os.path.join(ROOT, "LICENSE"), out_dir)

    for name in APP_FILES:
        target = os.path.join(out_dir, "src", name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        source = os.path.join(ROOT, "src", name)
        if os.path.exists(source):
            shutil.copy(source, target)
        else:
            open(target, "w").close()     # package marker

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).half()
    model.save_pretrained(os.path.join(out_dir, "model"))
    for name in MODEL_FILES:
        shutil.copy(os.path.join(model_dir, name), os.path.join(out_dir, "model"))


if __name__ == "__main__":
    build(*sys.argv[1:])

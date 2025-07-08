# src/utils.py
import os
from transformers import PreTrainedModel


def save_model(model: PreTrainedModel, tokenizer=None, path="artifacts/model"):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to {path}")


def load_model(path="artifacts/model"):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(path)

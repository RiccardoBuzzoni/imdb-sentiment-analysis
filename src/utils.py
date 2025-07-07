import torch
import os

def save_model(model, path="artifacts/model"):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"Model saved to {path}")

def load_model(path="artifacts/model"):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(path)
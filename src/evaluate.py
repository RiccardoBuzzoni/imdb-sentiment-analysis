from transformers import pipeline
from src.utils import load_model
import pandas as pd

def main():
    model = load_model()
    classifier = pipeline("text-classification", model=model, tokenizer="prajjwal1/bert-tiny")

    # Usa il sample CSV creato durante il training
    df = pd.read_csv("artifacts/imdb_sample.csv")
    sample_text = df.iloc[0]["text"]

    result = classifier(sample_text)
    print(f"Prediction for first sample: {result}")
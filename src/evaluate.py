from transformers import pipeline
from src.utils import load_model
import pandas as pd


def main():
    print("Carico il modello...")  # Debug
    model = load_model()

    print("Carico il tokenizer...")  # Debug
    classifier = pipeline(
        "text-classification", model=model, tokenizer="prajjwal1/bert-tiny"
    )

    print("Leggo il dataset...")  # Debug
    df = pd.read_csv("artifacts/imdb_sample.csv")
    sample_text = df.iloc[0]["text"]

    print("Faccio la predizione...")  # Debug
    result = classifier(sample_text)
    print(f"Prediction for first sample: {result}")


if __name__ == "__main__":
    main()

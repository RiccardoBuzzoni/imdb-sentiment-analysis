import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from src.model import get_model, get_tokenizer
from src.utils import save_model
import pandas as pd
import os

def preprocess_dataset(ds, tokenizer, max_length=512):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    return ds.map(tokenize_function, batched=True)

def main():
    # Carica dataset
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"])
    df = df.head(20000)  # Ridotto a 20k sample

    # Salva info per EDA
    df.to_csv("artifacts/imdb_sample.csv", index=False)

    # Preprocessing
    tokenizer = get_tokenizer()
    small_dataset = load_dataset("csv", data_files="artifacts/imdb_sample.csv")
    tokenized_datasets = preprocess_dataset(small_dataset['train'], tokenizer)

    # Split train/test
    train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_testvalid['train']
    test_dataset = train_testvalid['test']

    # Modello
    model = get_model()

    # Training
    training_args = TrainingArguments(
        output_dir="artifacts/training_output",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    save_model(model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
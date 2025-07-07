from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "prajjwal1/bert-tiny"

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def get_model(num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
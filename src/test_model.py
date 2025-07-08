from src.utils import load_model
import os


def test_model_loading():
    # Verifica che il modello possa essere caricato correttamente
    if not os.path.exists("artifacts/model"):
        print("Modello non presente. Salvalo prima con train.py")
        assert False
    try:
        model = load_model()
        assert model is not None
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        assert False

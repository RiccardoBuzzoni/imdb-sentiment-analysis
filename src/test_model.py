import os
import pytest
from src.utils import load_model

def test_model_loading():
    # Verifica che il modello possa essere caricato correttamente
    if not os.path.exists("artifacts/model"):
        pytest.skip("Modello non trovato: esegui prima 'python -m src.train'")
    
    try:
        model = load_model()
        assert model is not None
    except Exception as e:
        assert False, f"Errore durante il caricamento del modello: {e}"
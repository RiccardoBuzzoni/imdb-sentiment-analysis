# IMDb Sentiment Analyzer
    Progetto per analizzare il sentiment delle recensioni del dataset IMDb utilizzando un modello BERT Tiny addestrato in locale e containerizzato con Docker.

# Struttura del progetto
    imdb-sentiment-analyzer/
    ├── README.md
    ├── .gitignore
    ├── requirements.txt
    ├── setup.py
    ├── Dockerfile              # Definizione dell’immagine Docker
    ├── docker-compose.yml      # Configurazione multi-container opzionale
    ├── src/                    # Codice sorgente principale del progetto: training, valutazione, definizione del modello e utilità
    │   ├── __init__.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── model.py
    │   └── utils.py
    ├── notebooks/              # Jupyter Notebook per l’EDA (Exploratory Data Analysis)
    │   └── eda.ipynb
    └── .github/                # Configurazioni per CI/CD tramite GitHub Actions
        └── workflows/
            └── ci-cd.yml

# Obbiettivo del progetto
    - Analizzare le recensioni del dataset IMDb Movie Reviews
    - Addestrare un modello di classificazione basato su BERT Tiny
    - Eseguire una valutazione accurata del modello
    - Containerizzare l’applicazione con Docker
    - Implementare un sistema di CI/CD tramite GitHub Actions

# Modello Neurale Utilizzato
    - Modello : prajjwal1/bert-tiny
    - Motivazione : È una versione ridotta di BERT, ideale per risorse limitate come CPU AMD o GPU entry-level.
    - Framework : Hugging Face Transformers + PyTorch
    - Task : Classificazione binaria (positivo/negativo)

# Dataset scelto
    - Nome : IMDb Movie Reviews
    - Fonte : Hugging Face Datasets
    - Descrizione : Dataset di 50.000 recensioni cinematografiche etichettate come positive o negative
    - Quantità usata nel progetto : 20.000 campioni (ridotto per motivi computazionali)
    - Formato : JSON/csv (processato durante il training)

# Tecnologie utilizzate
    Linguaggio: Python 3.10+
    Ambiente: venv / pip
    Package: setuptools
    Git & Repo: GitHub
    Container: Docker
    CI/CD: GitHub Actions
    Linting: flake8
    Testing: pytest
    EDA: Pandas, Seaborn, Matplotlib, WordCloud

# EDA (Analisi Esplorativa)
    Il notebook notebooks/eda.ipynb contiene l’analisi esplorativa del dataset IMDb, tra cui:
        - Distribuzione delle etichette (pos, neg)
        - Distribuzione della lunghezza dei testi
        - Parole più frequenti
        - Visualizzazione tramite word cloud e grafici

    Il file CSV `artifacts/imdb_sample.csv` viene generato automaticamente durante il training e può essere usato per l’EDA.

# Dockerizzazione
    Il progetto è completamente containerizzato.

    Comandi principali:
        # Build dell'immagine Docker
        docker build -t imdb-sentiment-analysis .

        # Esecuzione del training in Docker
        docker run --rm -v $(pwd)/artifacts:/app/artifacts imdb-sentiment-analysis python src/train.py

        # Esecuzione del modello addestrato
        docker run --rm -v $(pwd)/artifacts:/app/artifacts imdb-sentiment-analysis python src/evaluate.py

# CI/CD con GitHub Actions
    Il file .github/workflows/ci-cd.yml implementa un workflow automatizzato che:
        1. Esegue il linting del codice (flake8)
        2. Avvia i test unitari (pytest)
        3. Builda l’immagine Docker

# Come usare il progetto
    1. Clona il repository
        git clone https://github.com/ riccardobuzzoni/imdb-sentiment-analyzer.git
        cd imdb-sentiment-analyzer

    2. Crea un ambiente virtuale
        python3 -m venv venv
        source venv/bin/activate # Linux/Mac
            oppure
        venv\Scripts\activate # Windows

        pip install -r requirements.txt

    3. Addestra il modello localmente
        python -m src.train

    4. Valuta il modello
        python -m src.evaluate
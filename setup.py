from setuptools import setup, find_packages

setup(
    name="imdb_sentiment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0"
    ],
    entry_points={
        "console_scripts": [
            "train-imdb=src.train:main",
            "eval-imdb=src.evaluate:main"
        ]
    }
)
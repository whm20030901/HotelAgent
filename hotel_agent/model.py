from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_dataset(csv_path: Path) -> Tuple[Iterable[str], Iterable[int]]:
    data = pd.read_csv(csv_path)
    if "label" not in data.columns or "review" not in data.columns:
        raise ValueError("CSV must contain 'label' and 'review' columns")
    return data["review"].astype(str), data["label"].astype(int)


@dataclass
class TrainingConfig:
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    max_iter: int = 500
    random_state: int = 42


class ModelTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.config.max_features,
                        ngram_range=self.config.ngram_range,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def train(self, csv_path: Path) -> Pipeline:
        texts, labels = load_dataset(csv_path)
        pipeline = self.build_pipeline()
        pipeline.fit(texts, labels)
        return pipeline

    def save(self, pipeline: Pipeline, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "sentiment_model.joblib"
        joblib.dump(pipeline, model_path)
        return model_path


class SentimentAnalyzer:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.pipeline: Pipeline = joblib.load(model_path)

    def predict(self, review: str) -> Tuple[int, float]:
        proba = self.pipeline.predict_proba([review])[0]
        label = int(proba.argmax())
        confidence = float(proba[label])
        return label, confidence

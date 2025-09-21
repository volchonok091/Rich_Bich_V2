"""Training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from richbich.config import ModelConfig
from richbich.utils.io import ensure_dir, save_json
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ModelTrainer:
    cfg: ModelConfig

    def _build_estimator(self):
        estimator = self.cfg.estimator.lower()
        params = self.cfg.params or {}
        if estimator == "random_forest":
            return RandomForestClassifier(random_state=self.cfg.random_state, **params)
        if estimator == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.cfg.random_state, **params)
        if estimator == "logistic_regression":
            return LogisticRegression(random_state=self.cfg.random_state, max_iter=1000, **params)
        raise ValueError(f"Unsupported estimator '{self.cfg.estimator}'")

    def _split(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        y = dataset["target_direction"]
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in {"target_direction", "target_return"}]
        X = dataset[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    def train(self, dataset: pd.DataFrame) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self._split(dataset)
        estimator = self._build_estimator()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        LOGGER.info("Fitting %s on %d samples", estimator.__class__.__name__, X_train.shape[0])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            y_proba = None
            roc_auc = float("nan")

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(report_dict["accuracy"]),
            "precision_up": float(report_dict["1"]["precision"]),
            "recall_up": float(report_dict["1"]["recall"]),
            "f1_up": float(report_dict["1"]["f1-score"]),
            "roc_auc": roc_auc,
        }
        LOGGER.info("Validation metrics: %s", metrics)

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_path = ensure_dir(Path(self.cfg.model_dir)) / f"{self.cfg.estimator}_{timestamp}.joblib"
        joblib.dump(pipeline, model_path)
        report_path = model_path.with_suffix(".metrics.json")
        save_json(report_path, metrics)
        LOGGER.info("Saved model to %s", model_path)
        return metrics
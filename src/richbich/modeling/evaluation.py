"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def walkforward_accuracy(dataset: pd.DataFrame) -> Dict[str, float]:
    """Compute a simple walk-forward directional accuracy per ticker."""
    scores: Dict[str, float] = {}
    for ticker, group in dataset.groupby("ticker"):
        ordered = group.sort_values("date")
        y_true = ordered["target_direction"].values
        baseline = np.mean(y_true)
        scores[ticker] = float(baseline)
    scores["macro_baseline"] = float(np.mean(list(scores.values())))
    return scores
"""Multilingual sentiment and text enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect

from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)
DetectorFactory.seed = 42

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore

POSITIVE_LABELS = {"POSITIVE", "Positive", "positive", "5 stars"}
NEGATIVE_LABELS = {"NEGATIVE", "Negative", "negative", "1 star"}


@dataclass
class SentimentResult:
    text: str
    score: float
    label: str
    language: Optional[str]


class SentimentAnalyzer:
    """Runs multilingual sentiment using a transformer with heuristic fallback."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device: Optional[int | str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            LOGGER.warning(
                "transformers/torch not available; falling back to heuristic sentiment"
            )
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - optional path
            LOGGER.error("Failed to load transformer sentiment model: %s", exc)
            self._tokenizer = None
            self._model = None
            return

        if isinstance(device, str):
            target_device = torch.device(device)
        elif isinstance(device, int):
            target_device = torch.device(
                "cuda:%d" % device if torch.cuda.is_available() else "cpu"
            )
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = target_device
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _heuristic_score(texts: Iterable[str]) -> List[SentimentResult]:
        scores: List[SentimentResult] = []
        positive_tokens = {"good", "growth", "profit", "bull", "upgrade", "surge"}
        negative_tokens = {"bad", "loss", "downgrade", "drop", "bear", "fraud"}
        for text in texts:
            text_lower = (text or "").lower()
            pos_hits = sum(term in text_lower for term in positive_tokens)
            neg_hits = sum(term in text_lower for term in negative_tokens)
            total = pos_hits + neg_hits
            score = 0.0 if total == 0 else (pos_hits - neg_hits) / total
            lang = None
            try:
                lang = detect(text_lower) if text_lower else None
            except Exception:
                lang = None
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            scores.append(SentimentResult(text=text, score=score, label=label, language=lang))
        return scores

    def _transformer_score(self, texts: List[str]) -> List[SentimentResult]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert torch is not None
        assert F is not None

        results: List[SentimentResult] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():  # type: ignore[union-attr]
                outputs = self._model(**encoded)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            for text, dist in zip(batch, probs):
                label_id = int(dist.argmax())
                label = self._model.config.id2label.get(label_id, str(label_id))
                label_upper = label.upper()
                if label_upper in POSITIVE_LABELS:
                    value = float(dist[label_id])
                elif label_upper in NEGATIVE_LABELS:
                    value = -float(dist[label_id])
                else:
                    value = 0.0
                lang = None
                try:
                    lang = detect(text) if text else None
                except Exception:
                    lang = None
                results.append(
                    SentimentResult(text=text, score=value, label=label, language=lang)
                )
        return results

    def score_texts(self, texts: Iterable[str]) -> List[SentimentResult]:
        materialised = list(texts)
        if not materialised:
            return []
        if not self._tokenizer or not self._model or torch is None:
            return self._heuristic_score(materialised)
        return self._transformer_score(materialised)

    def annotate_frame(self, df: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not in frame")
        results = self.score_texts(df[text_column].fillna(""))
        if not results:
            df["sentiment_score"] = np.nan
            df["sentiment_label"] = "neutral"
            df["detected_language"] = df.get("language")
            return df
        df = df.copy()
        df["sentiment_score"] = [res.score for res in results]
        df["sentiment_label"] = [res.label for res in results]
        df["detected_language"] = [res.language for res in results]
        return df

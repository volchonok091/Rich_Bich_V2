"""Feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from richbich.config import FeatureConfig
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FeatureBuilder:
    cfg: FeatureConfig

    def build(self, prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        dataset = self._merge_features(prices, news)
        dataset = dataset.dropna(subset=["target_return"])  # drop rows without forward return
        dataset.sort_values(["ticker", "date"], inplace=True)
        return dataset.reset_index(drop=True)

    def build_for_inference(self, prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        """Assemble feature matrix without dropping rows lacking forward returns."""
        dataset = self._merge_features(prices, news)
        dataset.sort_values(["ticker", "date"], inplace=True)
        return dataset.reset_index(drop=True)

    def _merge_features(self, prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        price_features = self._prepare_price_features(prices)
        news_features = self._prepare_news_features(news)
        dataset = price_features.merge(news_features, how="left", on=["ticker", "date"])
        dataset.fillna(
            {
                "news_count": 0,
                "sentiment_mean": 0.0,
                "sentiment_std": 0.0,
                "sentiment_roll_mean": 0.0,
                "tone_mean": 0.0,
                "relevance_mean": 0.0,
            },
            inplace=True,
        )
        return dataset

    def _prepare_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
        frame.dropna(subset=["close", "volume"], inplace=True)

        frame["return_1d"] = frame.groupby("ticker")["close"].pct_change(periods=1)
        frame["return_5d"] = frame.groupby("ticker")["close"].pct_change(periods=5)
        frame["vol_roll_std"] = (
            frame.groupby("ticker")["return_1d"].rolling(self.cfg.price_rolling_days, min_periods=1).std().reset_index(level=0, drop=True)
        )
        forward = (
            frame.groupby("ticker")["close"].pct_change(periods=self.cfg.target_horizon_days).shift(-self.cfg.target_horizon_days)
        )
        frame["target_return"] = forward
        frame["target_direction"] = (frame["target_return"] > 0).astype(int)
        price_cols = [
            "ticker",
            "date",
            "close",
            "volume",
            "return_1d",
            "return_5d",
            "vol_roll_std",
            "target_return",
            "target_direction",
        ]
        return frame[price_cols]

    def _prepare_news_features(self, news: pd.DataFrame) -> pd.DataFrame:
        if news.empty:
            LOGGER.warning("News frame is empty; returning placeholder features")
            return pd.DataFrame({
                "ticker": [],
                "date": [],
                "news_count": [],
                "sentiment_mean": [],
                "sentiment_std": [],
                "sentiment_roll_mean": [],
                "tone_mean": [],
                "relevance_mean": [],
                "languages": [],
            })

        frame = news.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame.sort_values(["ticker", "date"], inplace=True)

        aggregations = {
            "sentiment_score": ["mean", "std"],
            "tone": "mean",
            "relevance": "mean",
        }
        grouped = frame.groupby(["ticker", "date"]).agg(aggregations)
        grouped.columns = ["sentiment_mean", "sentiment_std", "tone_mean", "relevance_mean"]
        grouped["news_count"] = frame.groupby(["ticker", "date"]).size()
        languages = frame.groupby(["ticker", "date"]).apply(
            lambda g: ",".join(sorted({lang for lang in g.get("detected_language", []) if isinstance(lang, str) and lang}))
        )
        grouped["languages"] = languages
        grouped.reset_index(inplace=True)

        grouped["sentiment_std"].replace({np.nan: 0.0}, inplace=True)

        grouped["sentiment_roll_mean"] = (
            grouped.groupby("ticker")["sentiment_mean"].rolling(self.cfg.sentiment_rolling_days, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        grouped = grouped[grouped["news_count"] >= self.cfg.min_news_per_day]
        return grouped
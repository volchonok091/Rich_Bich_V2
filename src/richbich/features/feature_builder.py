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
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            dataset[numeric_cols] = dataset[numeric_cols].fillna(0.0)
        dataset.sort_values(["ticker", "date"], inplace=True)
        return dataset.reset_index(drop=True)

    def build_for_inference(self, prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        """Assemble feature matrix without dropping rows lacking forward returns."""
        dataset = self._merge_features(prices, news)
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            dataset[numeric_cols] = dataset[numeric_cols].fillna(0.0)
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
                "sentiment_median": 0.0,
                "sentiment_pos_ratio": 0.0,
                "sentiment_neg_ratio": 0.0,
                "sentiment_extreme_ratio": 0.0,
                "sentiment_roll_mean": 0.0,
                "tone_mean": 0.0,
                "relevance_mean": 0.0,
                "news_impact_mean": 0.0,
                "languages": "",
                "languages_count": 0,
            },
            inplace=True,
        )
        return dataset

    def _prepare_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["close"] = pd.to_numeric(frame.get("Close", frame.get("close")), errors="coerce")
        frame["volume"] = pd.to_numeric(frame.get("Volume", frame.get("volume")), errors="coerce")
        frame.dropna(subset=["close", "volume"], inplace=True)

        frame["return_1d"] = frame.groupby("ticker")["close"].pct_change(periods=1)
        frame["return_5d"] = frame.groupby("ticker")["close"].pct_change(periods=5)

        short_window = max(3, self.cfg.price_rolling_days)
        long_window = max(short_window * 3, short_window + 5)
        frame["sma_short"] = frame.groupby("ticker")["close"].transform(lambda s: s.rolling(short_window, min_periods=1).mean())
        frame["sma_long"] = frame.groupby("ticker")["close"].transform(lambda s: s.rolling(long_window, min_periods=1).mean())
        frame["sma_ratio"] = (frame["sma_short"] / frame["sma_long"]).replace([np.inf, -np.inf], np.nan) - 1.0

        frame["vol_roll_std"] = (
            frame.groupby("ticker")["return_1d"].rolling(self.cfg.price_rolling_days, min_periods=1).std().reset_index(level=0, drop=True)
        )
        frame["volume_roll_mean"] = frame.groupby("ticker")["volume"].transform(lambda s: s.rolling(long_window, min_periods=1).mean())
        frame["volume_zscore"] = frame.groupby("ticker")["volume"].transform(
            lambda s: (s - s.rolling(long_window, min_periods=2).mean()) / s.rolling(long_window, min_periods=2).std()
        )

        forward = (
            frame.groupby("ticker")["close"].pct_change(periods=self.cfg.target_horizon_days).shift(-self.cfg.target_horizon_days)
        )
        frame["target_return"] = forward
        frame["target_direction"] = (frame["target_return"] > 0).astype(int)

        for col in ["return_1d", "return_5d", "vol_roll_std", "sma_ratio", "volume_zscore"]:
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["volume_roll_mean"] = frame["volume_roll_mean"].fillna(frame["volume"])

        price_cols = [
            "ticker",
            "date",
            "close",
            "volume",
            "return_1d",
            "return_5d",
            "vol_roll_std",
            "sma_ratio",
            "volume_roll_mean",
            "volume_zscore",
            "target_return",
            "target_direction",
        ]
        return frame[price_cols]

    def _prepare_news_features(self, news: pd.DataFrame) -> pd.DataFrame:
        if news.empty:
            LOGGER.warning("News frame is empty; returning placeholder features")
            return pd.DataFrame(
                {
                    "ticker": [],
                    "date": [],
                    "news_count": [],
                    "sentiment_mean": [],
                    "sentiment_std": [],
                    "sentiment_median": [],
                    "sentiment_pos_ratio": [],
                    "sentiment_neg_ratio": [],
                    "sentiment_extreme_ratio": [],
                    "sentiment_roll_mean": [],
                    "tone_mean": [],
                    "relevance_mean": [],
                    "news_impact_mean": [],
                    "languages": [],
                    "languages_count": [],
                }
            )

        frame = news.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame.sort_values(["ticker", "date"], inplace=True)
        frame["sentiment_score"] = pd.to_numeric(frame.get("sentiment_score", frame.get("tone", 0.0)), errors="coerce").fillna(0.0)
        frame["tone"] = pd.to_numeric(frame.get("tone", 0.0), errors="coerce").fillna(0.0)
        frame["relevance"] = pd.to_numeric(frame.get("relevance", 0.0), errors="coerce").fillna(0.0)
        frame["news_impact"] = frame["tone"] * frame["relevance"]

        group_keys = ["ticker", "date"]
        base_stats = frame.groupby(group_keys)["sentiment_score"].agg(["mean", "std", "median"]).rename(
            columns={"mean": "sentiment_mean", "std": "sentiment_std", "median": "sentiment_median"}
        )
        base_stats["sentiment_std"] = base_stats["sentiment_std"].fillna(0.0)

        ratio_stats = frame.groupby(group_keys).agg(
            sentiment_pos_ratio=("sentiment_score", lambda s: float((s > 0.05).sum()) / float(len(s)) if len(s) else 0.0),
            sentiment_neg_ratio=("sentiment_score", lambda s: float((s < -0.05).sum()) / float(len(s)) if len(s) else 0.0),
            sentiment_extreme_ratio=("sentiment_score", lambda s: float((s.abs() >= 0.5).sum()) / float(len(s)) if len(s) else 0.0),
        ).reset_index()

        news_count = frame.groupby(group_keys).size().rename("news_count").reset_index()
        tone_relevance = frame.groupby(group_keys)[["tone", "relevance", "news_impact"]].mean().rename(columns={"tone": "tone_mean", "relevance": "relevance_mean", "news_impact": "news_impact_mean"}).reset_index()

        language_cols = [col for col in ("detected_language", "language") if col in frame.columns]
        if language_cols:
            lang_lists = frame.groupby(group_keys)[language_cols].agg(
                lambda col: [str(val).strip() for val in col.dropna() if str(val).strip()]
            ).reset_index()

            def _collapse(row: pd.Series) -> pd.Series:
                codes: set[str] = set()
                for col in language_cols:
                    for value in row.get(col, []) or []:
                        if value:
                            codes.add(value)
                joined = ",".join(sorted(codes))
                return pd.Series({"languages": joined, "languages_count": len(codes)})

            lang_stats = lang_lists.apply(_collapse, axis=1)
            languages = pd.concat([lang_lists[group_keys], lang_stats], axis=1)
        else:
            languages = pd.DataFrame({"ticker": [], "date": [], "languages": [], "languages_count": []})

        aggregated = base_stats.reset_index()
        aggregated = aggregated.merge(ratio_stats, on=group_keys, how="left")
        aggregated = aggregated.merge(news_count, on=group_keys, how="left")
        aggregated = aggregated.merge(tone_relevance, on=group_keys, how="left")
        aggregated = aggregated.merge(languages, on=group_keys, how="left")

        aggregated["sentiment_roll_mean"] = (
            aggregated.groupby("ticker")["sentiment_mean"].rolling(self.cfg.sentiment_rolling_days, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        aggregated["sentiment_roll_mean"] = aggregated["sentiment_roll_mean"].fillna(0.0)
        aggregated = aggregated[aggregated["news_count"] >= self.cfg.min_news_per_day]
        return aggregated




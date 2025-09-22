"""End-to-end training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from richbich.config import TrainingConfig
from richbich.data.news_fetcher import fetch_all_news
from richbich.data.price_loader import download_prices
from richbich.utils.aliases import register_aliases, query_fallback_terms
from richbich.features.feature_builder import FeatureBuilder
from richbich.modeling.trainer import ModelTrainer
from richbich.nlp.sentiment import SentimentAnalyzer
from richbich.utils.io import ensure_dir
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.sentiment = SentimentAnalyzer()
        self.feature_builder = FeatureBuilder(cfg.features)
        self.trainer = ModelTrainer(cfg.model)

    def prepare_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices = download_prices(
            self.cfg.data.tickers,
            self.cfg.data.start_date,
            self.cfg.data.end_date,
            interval=self.cfg.data.interval,
            cache_dir=self.cfg.data.price_cache_dir,
        )
        news = fetch_all_news(
            self.cfg.news.per_ticker_queries,
            self.cfg.news.start_date,
            self.cfg.news.end_date,
            batch_days=self.cfg.news.batch_days,
            max_records=self.cfg.news.max_records,
            sleep_seconds=self.cfg.news.sleep_seconds,
            cache_dir=self.cfg.news.cache_dir,
        )

        annotated_news = self.sentiment.annotate_frame(news)
        features = self.feature_builder.build(prices, annotated_news)

        processed_dir = ensure_dir(self.cfg.features.processed_dir)
        prices_path = processed_dir / f"{self.cfg.experiment_name}_prices.csv"
        news_path = processed_dir / f"{self.cfg.experiment_name}_news.csv"
        feature_path = processed_dir / f"{self.cfg.experiment_name}_features.csv"
        prices.to_csv(prices_path, index=False)
        annotated_news.to_csv(news_path, index=False)
        features.to_csv(feature_path, index=False)
        LOGGER.info("Persisted processed datasets to %s", processed_dir)

        return prices, annotated_news, features

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        prices, news, features = self.prepare_datasets()
        metrics = self.trainer.train(features)
        return prices, news, features, metrics


"""Configuration objects and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"
    price_cache_dir: Path = Path("data/raw/prices")


@dataclass
class NewsConfig:
    per_ticker_queries: Dict[str, List[str]]
    start_date: str
    end_date: str
    max_records: int = 250
    batch_days: int = 7
    sleep_seconds: float = 1.0
    cache_dir: Path = Path("data/raw/news")

    def queries_for(self, ticker: str) -> List[str]:
        return self.per_ticker_queries.get(ticker.upper(), [])


@dataclass
class FeatureConfig:
    sentiment_rolling_days: int = 3
    price_rolling_days: int = 5
    target_horizon_days: int = 1
    min_news_per_day: int = 1
    processed_dir: Path = Path("data/processed")


@dataclass
class ModelConfig:
    estimator: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    params: Dict[str, Any] = field(default_factory=dict)
    model_dir: Path = Path("models")


@dataclass
class TrainingConfig:
    data: DataConfig
    news: NewsConfig
    features: FeatureConfig
    model: ModelConfig
    experiment_name: str = "richbich_run"


def _normalise_paths(cfg: TrainingConfig) -> None:
    cfg.data.price_cache_dir = Path(cfg.data.price_cache_dir)
    cfg.news.cache_dir = Path(cfg.news.cache_dir)
    cfg.features.processed_dir = Path(cfg.features.processed_dir)
    cfg.model.model_dir = Path(cfg.model.model_dir)


def load_config(path: Path | str) -> TrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    cfg = TrainingConfig(
        data=DataConfig(**raw["data"]),
        news=NewsConfig(**raw["news"]),
        features=FeatureConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        experiment_name=raw.get("experiment_name", path.stem),
    )
    _normalise_paths(cfg)
    return cfg
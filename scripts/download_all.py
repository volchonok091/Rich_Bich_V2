"""Download price and news data without training."""
from __future__ import annotations

import argparse

from richbich.config import load_config
from richbich.pipelines.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download market data and multilingual news")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = TrainingPipeline(cfg)
    pipeline.prepare_datasets()


if __name__ == "__main__":
    main()
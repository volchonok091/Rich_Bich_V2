"""Download price and news data without training."""
from __future__ import annotations

import argparse
import time

from richbich.utils.logging import get_logger

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
    LOGGER = get_logger(__name__)
    start_ts = time.perf_counter()
    pipeline.prepare_datasets()
    elapsed = time.perf_counter() - start_ts
    LOGGER.info("Download pipeline completed in %.2fs", elapsed)


if __name__ == "__main__":
    main()
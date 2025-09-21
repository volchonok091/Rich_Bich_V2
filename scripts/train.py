"""Train the RichBich model end-to-end."""
from __future__ import annotations

import argparse
import json

from richbich.config import load_config
from richbich.pipelines.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock direction forecaster")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument("--print-metrics", action="store_true", help="Print metrics to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = TrainingPipeline(cfg)
    _, _, _, metrics = pipeline.run()
    if args.print_metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
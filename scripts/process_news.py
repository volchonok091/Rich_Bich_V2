"""CLI for processing aggregated news feeds by company."""
from __future__ import annotations

import argparse

from richbich.data.news_processor import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process aggregated feeds into company news")
    parser.add_argument("--days", type=int, default=60, help="Number of days to keep (default: 60)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(days=args.days)


if __name__ == "__main__":
    main()

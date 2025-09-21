"""CLI for ingesting open news feeds."""
from __future__ import annotations

import argparse

from richbich.data.feed_ingestor import load_sources, ingest_all, persist_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest open news feeds into local cache")
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional list of source IDs to ingest (default: all from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = load_sources()
    if args.sources:
        sources = [s for s in sources if s.id in set(args.sources)]
    if not sources:
        print("No sources configured")
        return
    df = ingest_all(sources)
    persist_dataframe(df)


if __name__ == "__main__":
    main()

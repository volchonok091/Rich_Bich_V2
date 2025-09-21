"""Run the RichBich Telegram bot."""
from __future__ import annotations

import argparse
from pathlib import Path

from richbich.bot.telegram_bot import run_bot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RichBich Telegram bot")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--token", default=None, help="Telegram bot token (overrides env variable)")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit path to a trained model (.joblib). Latest model is used by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bot(token=args.token, config_path=Path(args.config), model_path=args.model_path)


if __name__ == "__main__":
    main()

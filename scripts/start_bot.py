"""Interactive launcher for the RichBich Telegram bot."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from richbich.bot.telegram_bot import run_bot


def prompt(message: str, *, secret: bool = False) -> str:
    if secret:
        return getpass.getpass(message).strip()
    return input(message).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive launcher for RichBich bot")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        token = prompt("Enter Telegram bot token: ")
    if not token:
        raise SystemExit("Telegram token is required.")
    os.environ["TELEGRAM_BOT_TOKEN"] = token

    if not os.environ.get("OPENAI_API_KEY"):
        key = prompt("OpenAI API key (optional, press Enter to skip): ", secret=True)
        if key:
            os.environ["OPENAI_API_KEY"] = key

    run_bot(token=token, config_path=Path(args.config), model_path=args.model_path)


if __name__ == "__main__":
    main()

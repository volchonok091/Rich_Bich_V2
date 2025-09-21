# RichBich V2 - Multilingual Market Intelligence Suite

RichBich V2 ingests prices and news in English, German, Arabic, and Russian, extracts sentiment-aware features, trains a forecasting model, and serves the results through a Telegram bot or a desktop launcher. This repository contains the full data pipeline, model training code, LLM integration, and a GUI for one-click operations.

## Highlights

- Global news coverage via configurable RSS feeds and GDELT Doc API queries.
- Extended ticker universe (NASDAQ, NYSE, MOEX, crypto pairs, NAS100 index) maintained in `configs/training.yaml`.
- Feature engineering and modeling with reusable pipelines and persisted artifacts.
- Telegram bot that understands free-form queries in multiple languages, summarises news, and explains the forecast.
- Launcher GUI (`scripts/launcher_gui.py`) to run training, tweak LLM providers, and operate the bot without touching the console.
- Pluggable LLM client supporting OpenAI-compatible endpoints, Azure, and local Ollama with heuristic fallback.

## Project Layout

```
rich_bich/
|-- configs/        # YAML configs, company index, RSS source list
|-- data/           # Cached prices/news (gitignored)
|-- models/         # Trained pipelines and metrics reports
|-- scripts/        # download_all.py, train.py, start_bot.py, launcher_gui.py
`-- src/richbich/   # Package modules (data, features, modeling, services, bot)
```

## Quickstart

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download prices & news
python scripts\download_all.py --config configs\training.yaml

# 3. Train model
python scripts\train.py --config configs\training.yaml --print-metrics

# 4. Run Telegram bot (CLI)
$env:TELEGRAM_BOT_TOKEN="<your-token>"
python scripts\start_bot.py --config configs\training.yaml
```

> Tip: ensure every ticker listed under `data.tickers` also has query terms in `news.per_ticker_queries`; otherwise news fetching will skip it.

## Launcher GUI

```
python scripts\launcher_gui.py
```

The window lets you:

- paste the Telegram bot token;
- choose the LLM provider (`openai`, `openai-compatible`, `azure`, `ollama`, `none`) and define API key / base URL / model ID;
- start or stop the bot with one click;
- trigger a 90-day "quick train" run to refresh the model fast;
- tail a unified stdout/stderr log from bot and training processes.

All values are injected into environment variables for the spawned processes (`TELEGRAM_BOT_TOKEN`, `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`).

## Configuration Overview

### `configs/training.yaml`

- `data.tickers` - tickers the pipeline should learn and the bot will serve.
- `news.per_ticker_queries` - search keywords passed to GDELT for each ticker (add local-language aliases such as "intel" for Russian queries).
- `news.batch_days` - window size for GDELT requests; larger windows respond faster.
- `features` - rolling windows for sentiment/price and target horizon.
- `model` - estimator choice (`random_forest`, `gradient_boosting`, `logistic_regression`) and hyperparameters.

### `configs/news_sources.yaml`

Curated RSS/Atom feeds spanning Reuters, CNBC, MarketWatch, FAZ, Al Jazeera, Interfax, and more. Extend this list to add public sources.

### `configs/company_index.csv`

Lookup table of symbols, companies, exchanges, and aliases. The Telegram bot uses it to match free text (e.g., "intel", "tinkoff") to tickers. Keep it in sync with `data.tickers`.

## Data & Model Pipeline

1. `scripts/download_all.py` orchestrates price downloads (`price_loader`) and news ingestion (`news_fetcher`).
2. `FeatureBuilder` merges prices, news, and sentiment into model-ready features.
3. `scripts/train.py` runs the training pipeline, evaluates metrics, and writes artifacts to `models/`.
4. `scripts/start_bot.py` (or the GUI) loads the latest pipeline and serves predictions/explanations through Telegram.

## Telegram Bot Usage

Once running, the bot supports:

- `/predict <ticker>` - probability of an up move and model signal.
- `/news <ticker> [days]` - latest cached headlines with sentiment scores.
- Free-form questions in supported languages - the bot detects the company, derives the requested time window, summarises multilingual news, and generates a forecast explanation (with LLM when configured).

Provide `LLM_API_KEY` (or legacy `OPENAI_API_KEY`) to enable LLM narratives. Without a key, the bot falls back to deterministic heuristics.

## Diagnostics

```powershell
# Syntax check
python -m compileall src

# Verify company registry loading
$env:PYTHONPATH='src'
python - <<'PY'
from richbich.services import company_registry
print(len(list(company_registry.iter_records())))
PY
```

## Deployment Tips

- Store secrets (`TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, etc.) in a `.env` that is excluded from git.
- Schedule periodic runs of `download_all.py` and `train.py` (Task Scheduler / cron) to keep the model fresh.
- Use the launcher on analyst desktops for zero-CLI operations.

## License

Provided "as is". Check the terms of external services you rely on (GDELT, RSS feeds, LLM providers) and comply with their usage limits.

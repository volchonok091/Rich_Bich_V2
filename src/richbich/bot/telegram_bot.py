"""Telegram bot integration for RichBich."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from richbich.config import TrainingConfig, load_config
from richbich.data.news_fetcher import fetch_news_for_ticker
from richbich.data.price_loader import download_prices
from richbich.features.feature_builder import FeatureBuilder
from richbich.nlp.sentiment import SentimentAnalyzer
from richbich.services.company_lookup import CompanyMatch, search_company
from richbich.services import company_registry
from richbich.services.llm_client import LLMClient
from richbich.services.summarizer import NewsSummarizer
from richbich.utils.io import ensure_dir
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ForecastResult:
    ticker: str
    date: str
    probability_up: float
    direction: str
    close: float


@dataclass
class AdvisoryReport:
    symbol: str
    company: str
    probability_up: Optional[float]
    direction: Optional[str]
    latest_close: Optional[float]
    price_change_30d: Optional[float]
    sentiment_mean: Optional[float]
    news_count: int
    summary: str
    explanation: str
    highlights: List[dict]


class ForecastService:
    """Provides predictions, summaries, and explanations for tickers."""

    def __init__(self, cfg: TrainingConfig, model_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.features_path = cfg.features.processed_dir / f"{cfg.experiment_name}_features.csv"
        self.news_path = cfg.features.processed_dir / f"{cfg.experiment_name}_news.csv"
        self._features = self._load_features()
        self._news = self._load_news()
        self._pipeline = self._load_model(model_path)
        self._feature_columns = self._determine_feature_columns()
        self._tickers = {t.upper() for t in cfg.data.tickers}
        self._company_records = list(company_registry.iter_records())
        self._local_feed_path = Path("data/processed/news_feed.jsonl")
        self._sentiment = SentimentAnalyzer()
        self._feature_builder = FeatureBuilder(cfg.features)
        self._summariser = NewsSummarizer()
        self._llm = LLMClient()
        self._price_cache_dir = ensure_dir(Path(cfg.data.price_cache_dir))
        self._adhoc_news_dir = ensure_dir(Path(cfg.news.cache_dir) / "adhoc")

    def _load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {self.features_path}. Run the training pipeline first."
            )
        df = pd.read_csv(self.features_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        return df

    def _load_news(self) -> pd.DataFrame:
        if not self.news_path.exists():
            LOGGER.warning("News file %s missing; news command will be limited", self.news_path)
            return pd.DataFrame()
        df = pd.read_csv(self.news_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
        if "seen_datetime" in df.columns:
            df["seen_datetime"] = pd.to_datetime(df["seen_datetime"], errors="coerce")
        return df

    def _latest_model_path(self) -> Path:
        model_dir = Path(self.cfg.model.model_dir)
        candidates = sorted(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No model artefacts found in {model_dir}. Train the model first.")
        return candidates[0]

    def _load_model(self, model_path: Optional[Path]) -> object:
        path = Path(model_path) if model_path else self._latest_model_path()
        LOGGER.info("Loading inference pipeline from %s", path)
        pipeline = joblib.load(path)
        return pipeline

    def _determine_feature_columns(self) -> List[str]:
        numeric_cols = self._features.select_dtypes(include=["number"]).columns.tolist()
        return [col for col in numeric_cols if col not in {"target_return", "target_direction"}]

    def list_tickers(self) -> List[str]:
        return sorted(self._tickers)

    def _resolve_ticker(self, ticker: str) -> Optional[str]:
        symbol = ticker.upper()
        return symbol if symbol in self._tickers else None

    def latest_prediction(self, ticker: str) -> ForecastResult:
        symbol = self._resolve_ticker(ticker)
        if not symbol:
            raise ValueError(f"Ticker '{ticker}' is not tracked. Known tickers: {', '.join(self.list_tickers())}")
        subset = self._features[self._features["ticker"] == symbol].sort_values("date")
        if subset.empty:
            raise ValueError(f"No feature rows available for {symbol}. Run the pipeline to refresh data.")
        latest = subset.iloc[-1]
        features = latest[self._feature_columns].to_frame().T
        if hasattr(self._pipeline, "predict_proba"):
            proba_up = float(self._pipeline.predict_proba(features)[0][1])
        else:
            proba_up = float(self._pipeline.predict(features)[0])
        direction = "up" if proba_up >= 0.5 else "down"
        return ForecastResult(
            ticker=symbol,
            date=latest["date"].strftime("%Y-%m-%d"),
            probability_up=proba_up,
            direction=direction,
            close=float(latest.get("close", float("nan"))),
        )

    def recent_news(self, ticker: str, days: int = 3, limit: int = 5) -> List[dict]:
        if self._news.empty:
            return []
        symbol = self._resolve_ticker(ticker)
        if not symbol:
            raise ValueError(f"Ticker '{ticker}' is not tracked. Known tickers: {', '.join(self.list_tickers())}")
        news = self._news[self._news["ticker"] == symbol]
        if news.empty:
            return []
        cutoff = news["date"].max() - timedelta(days=days)
        order_col = "seen_datetime" if "seen_datetime" in news.columns else "date"
        recent = news[news["date"] >= cutoff].sort_values(order_col, ascending=False).head(limit)
        records: List[dict] = []
        for _, row in recent.iterrows():
            records.append(
                {
                    "date": row.get("date").strftime("%Y-%m-%d") if pd.notna(row.get("date")) else "",
                    "title": row.get("title", "(no title)"),
                    "url": row.get("url", ""),
                    "sentiment": float(row.get("sentiment_score", 0.0)),
                    "source": row.get("source", ""),
                    "language": row.get("detected_language", row.get("language", "")),
                }
            )
        return records

    def match_companies(self, query: str, limit: int = 5) -> List[CompanyMatch]:
        records = self._company_records
        if not query or not query.strip():
            return []
        cleaned = query.strip()
        norm = company_registry.normalise_query(cleaned)
        matches: List[CompanyMatch] = []
        direct = company_registry.find_by_symbol(cleaned)
        if direct:
            matches.append(
                CompanyMatch(direct.symbol, direct.company, direct.company, direct.exchange, 1.0)
            )
        for record in records:
            if any(match.symbol == record.symbol for match in matches):
                continue
            if norm in record.aliases:
                matches.append(
                    CompanyMatch(record.symbol, record.company, record.company, record.exchange, 0.9)
                )
        if len(matches) < limit:
            aliases = {alias: record for record in records for alias in record.aliases}
            for alias, record in aliases.items():
                if alias.startswith(norm) or norm in alias:
                    if any(match.symbol == record.symbol for match in matches):
                        continue
                    matches.append(
                        CompanyMatch(record.symbol, record.company, record.company, record.exchange, 0.75)
                    )
        if not matches:
            compact = cleaned.replace(" ", "")
            upper = compact.upper()
            if compact == upper and upper.isalnum() and 1 <= len(upper) <= 6:
                matches.append(CompanyMatch(upper, upper, upper, None, None))
        return matches[:limit]

    def build_report_from_match(
        self,
        match: CompanyMatch,
        *,
        original_query: Optional[str] = None,
    ) -> AdvisoryReport:
        return self._build_report(match, original_query=original_query)

    def analyse_free_text(self, text: str) -> AdvisoryReport:
        matches = self.match_companies(text)
        if not matches:
            raise ValueError("Не удалось распознать компанию по запросу. Уточни название или тикер.")
        return self._build_report(matches[0], original_query=text)

    def _build_report(
        self,
        match: CompanyMatch,
        *,
        original_query: Optional[str] = None,
    ) -> AdvisoryReport:
        symbol = match.symbol.upper()
        company_name = match.shortname or match.longname or symbol
        now = datetime.utcnow()
        price_start = (now - timedelta(days=180)).date().isoformat()
        price_end = (now + timedelta(days=1)).date().isoformat()
        try:
            prices = download_prices(
                [symbol],
                price_start,
                price_end,
                interval=self.cfg.data.interval,
                cache_dir=self._price_cache_dir,
            )
        except ValueError as exc:
            LOGGER.error("Price download failed for %s: %s", symbol, exc)
            prices = pd.DataFrame()

        news_start = (now - timedelta(days=30)).date().isoformat()
        news_end = now.date().isoformat()
        query_terms = [term for term in {match.shortname, match.longname, symbol} if term]
        news_df = pd.DataFrame()
        if query_terms:
            try:
                news_df = fetch_news_for_ticker(
                    symbol,
                    query_terms,
                    news_start,
                    news_end,
                    batch_days=self.cfg.news.batch_days,
                    max_records=self.cfg.news.max_records,
                    sleep_seconds=max(0.2, self.cfg.news.sleep_seconds),
                    cache_dir=self._adhoc_news_dir,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.error("Failed to fetch news for %s: %s", symbol, exc)
                news_df = pd.DataFrame()

        local_news = self._local_news_for(symbol, company_name)
        if not local_news.empty:
            if news_df.empty:
                news_df = local_news
            else:
                news_df = pd.concat([news_df, local_news], ignore_index=True)
                news_df.drop_duplicates(subset=["title", "url"], inplace=True)

        annotated_news = self._annotate_news(news_df)
        feature_frame = self._build_features_for_inference(prices, annotated_news, symbol)
        probability, direction = self._score_probability(feature_frame)
        latest_close, price_change = self._extract_price_metrics(feature_frame)
        sentiment_mean = self._extract_sentiment(annotated_news)
        summary_result = self._summariser.summarise(self._collect_summary_texts(annotated_news))
        highlights = self._build_highlights(annotated_news)
        explanation = self._compose_explanation(
            company_name=company_name,
            symbol=symbol,
            probability=probability,
            direction=direction,
            price_change=price_change,
            sentiment_mean=sentiment_mean,
            summary=summary_result.text,
            highlights=highlights,
            original_query=original_query,
        )

        return AdvisoryReport(
            symbol=symbol,
            company=company_name,
            probability_up=probability,
            direction=direction,
            latest_close=latest_close,
            price_change_30d=price_change,
            sentiment_mean=sentiment_mean,
            news_count=len(annotated_news) if not annotated_news.empty else 0,
            summary=summary_result.text,
            explanation=explanation,
            highlights=highlights,
        )

    def _local_feed_df(self) -> pd.DataFrame:
        if not self._local_feed_path.exists():
            return pd.DataFrame()
        df = pd.read_json(self._local_feed_path, lines=True)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)
        df.dropna(subset=["published_at"], inplace=True)
        df["title"] = df["title"].fillna("")
        df["summary"] = df["summary"].fillna("")
        df["combined_norm"] = (
            df["title"].map(company_registry.normalise_query)
            + " "
            + df["summary"].map(company_registry.normalise_query)
        )
        return df

    def _local_news_for(self, symbol: str, company_name: str, days: int = 30) -> pd.DataFrame:
        df = self._local_feed_df()
        if df.empty:
            return pd.DataFrame()
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["published_at"] >= cutoff]
        alias_norms = set(company_registry.symbol_aliases(symbol))
        alias_norms.add(company_registry.normalise_query(company_name))
        alias_norms.add(company_registry.normalise_query(symbol))
        alias_norms = {alias for alias in alias_norms if alias}
        if not alias_norms:
            return pd.DataFrame()
        mask = df["combined_norm"].apply(lambda text: any(alias in text for alias in alias_norms))
        subset = df[mask]
        if subset.empty:
            return pd.DataFrame()
        subset = subset.copy()
        subset["ticker"] = symbol
        subset["seen_datetime"] = subset["published_at"]
        subset["date"] = subset["published_at"].dt.date
        subset["full_text"] = subset["summary"]
        subset["url"] = subset["link"]
        subset["source"] = subset["source_name"]
        subset["language"] = subset.get("language").fillna("")
        columns = ["ticker", "seen_datetime", "date", "title", "full_text", "url", "language", "source"]
        return subset[columns]

    def _annotate_news(self, news: pd.DataFrame) -> pd.DataFrame:
        if news.empty:
            return news
        try:
            return self._sentiment.annotate_frame(news)
        except Exception as exc:
            LOGGER.error("Failed to annotate news sentiment: %s", exc)
            return news

    def _build_features_for_inference(
        self,
        prices: pd.DataFrame,
        news: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()
        if not news.empty and "ticker" not in news.columns:
            news = news.copy()
            news["ticker"] = symbol
        try:
            features = self._feature_builder.build_for_inference(prices, news)
        except Exception as exc:
            LOGGER.error("Failed to build features for %s: %s", symbol, exc)
            return pd.DataFrame()
        return features[features["ticker"] == symbol]

    def _score_probability(self, feature_frame: pd.DataFrame) -> tuple[Optional[float], Optional[str]]:
        if feature_frame.empty:
            return None, None
        latest = feature_frame.sort_values("date").iloc[[-1]]
        matrix = latest[self._feature_columns].reindex(columns=self._feature_columns, fill_value=0.0)
        if hasattr(self._pipeline, "predict_proba"):
            proba = float(self._pipeline.predict_proba(matrix)[0][1])
        else:
            proba = float(self._pipeline.predict(matrix)[0])
        direction = "рост" if proba >= 0.5 else "снижение"
        return proba, direction

    @staticmethod
    def _extract_price_metrics(feature_frame: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        if feature_frame.empty:
            return None, None
        ordered = feature_frame.sort_values("date")
        closes = ordered["close"].dropna()
        if closes.empty:
            return None, None
        latest_close = float(closes.iloc[-1])
        change = None
        if len(closes) > 21:
            base = closes.iloc[-22]
            if base and not math.isclose(base, 0.0):
                change = float((latest_close / base) - 1.0)
        return latest_close, change

    @staticmethod
    def _extract_sentiment(news: pd.DataFrame) -> Optional[float]:
        if news.empty or "sentiment_score" not in news.columns:
            return None
        series = news["sentiment_score"].dropna()
        if series.empty:
            return None
        return float(series.mean())

    @staticmethod
    def _collect_summary_texts(news: pd.DataFrame) -> List[str]:
        if news.empty:
            return []
        order_col = "seen_datetime" if "seen_datetime" in news.columns else "date"
        rows = news.sort_values(order_col, ascending=False).head(12)
        docs = []
        for _, row in rows.iterrows():
            title = row.get("title", "")
            body = row.get("full_text", "")
            docs.append(f"{title}. {body}")
        return docs

    @staticmethod
    def _build_highlights(news: pd.DataFrame, limit: int = 5) -> List[dict]:
        if news.empty:
            return []
        order_col = "seen_datetime" if "seen_datetime" in news.columns else "date"
        rows = news.sort_values(order_col, ascending=False).head(limit)
        highlights = []
        for _, row in rows.iterrows():
            highlights.append(
                {
                    "date": row.get("date").strftime("%Y-%m-%d") if pd.notna(row.get("date")) else "",
                    "title": row.get("title", "(без названия)"),
                    "url": row.get("url", ""),
                    "source": row.get("source", row.get("language", "")),
                    "sentiment": float(row.get("sentiment_score", 0.0)),
                }
            )
        return highlights

    def _compose_explanation(
        self,
        *,
        company_name: str,
        symbol: str,
        probability: Optional[float],
        direction: Optional[str],
        price_change: Optional[float],
        sentiment_mean: Optional[float],
        summary: str,
        highlights: List[dict],
        original_query: Optional[str],
    ) -> str:
        probability_text = f"{probability:.1%}" if probability is not None else "нет данных"
        direction_text = direction or "нет данных"
        price_change_text = f"{price_change:+.1%}" if price_change is not None else "нет данных"
        sentiment_text = f"{sentiment_mean:+.2f}" if sentiment_mean is not None else "нет данных"
        summary_text = summary or "нет свежих новостей"

        if self._llm.available:
            highlight_lines = []
            for item in highlights:
                line = (
                    f"- {item.get('date', '')} | {item.get('source', '')}: {item.get('title', '')} "
                    f"(sentiment {item.get('sentiment', 0.0):+.2f})"
                ).strip()
                if item.get('url'):
                    line += f"\n  {item['url']}"
                highlight_lines.append(line)
            prompt_parts = [
                f"Клиент спросил: {original_query or company_name}.",
                f"Компания: {company_name} ({symbol}).",
                f"Вероятность роста по модели: {probability_text}.",
                f"Прогноз направления: {direction_text}.",
                f"Изменение цены за 30 дней: {price_change_text}.",
                f"Средний новостной сентимент: {sentiment_text}.",
                f"Выжимка новостей: {summary_text}.",
                "Ключевые заголовки:",
            ]
            prompt_parts.extend(highlight_lines or ["нет выделенных материалов."])
            prompt_parts.append(
                "Сформулируй ответ на русском языке, как брокер-консультант. До 4 предложений, без жаргона,"
                " с напоминанием о рисках и отсутствии гарантий."
            )
            user_prompt = "\n".join(prompt_parts)
            system_prompt = (
                "Ты финансовый консультант. Объясняй простыми словами и всегда упоминай риски."
            )
            llm_response = self._llm.generate(system_prompt, user_prompt)
            if llm_response:
                return llm_response.text

        reasons: List[str] = []
        if probability is not None:
            if probability >= 0.65:
                reasons.append(f"Модель видит высокую вероятность роста (≈{probability:.0%}).")
            elif probability <= 0.35:
                reasons.append(f"Модель предполагает снижение (вероятность роста около {probability:.0%}).")
            else:
                reasons.append(f"Модель оценивает шансы роста примерно в {probability:.0%}, ситуация неопределённая.")
        if price_change is not None:
            if price_change >= 0:
                reasons.append(f"За последний месяц цена выросла примерно на {price_change:.1%}.")
            else:
                reasons.append(f"За последний месяц цена снизилась примерно на {price_change:.1%}.")
        if sentiment_mean is not None:
            if sentiment_mean > 0.05:
                reasons.append(f"Новостной фон скорее позитивный (средний сентимент {sentiment_mean:+.2f}).")
            elif sentiment_mean < -0.05:
                reasons.append(f"Новостной фон негативный (средний сентимент {sentiment_mean:+.2f}).")
            else:
                reasons.append("Новостной фон нейтральный.")
        if not reasons:
            reasons.append("Недостаточно данных — оцени риски и диверсифицируй портфель.")
        return " ".join(reasons)

    def build_messages(self, report: AdvisoryReport) -> List[str]:
        header_lines: List[str] = [
            f"Отчёт по компании {report.company} ({report.symbol})",
        ]
        if report.probability_up is not None:
            header_lines.append(f"Вероятность роста: {report.probability_up:.1%}")
        if report.direction:
            header_lines.append(f"Сигнал модели: {report.direction}")
        if report.latest_close is not None:
            header_lines.append(f"Последняя цена закрытия: {report.latest_close:,.2f} USD")
        if report.price_change_30d is not None:
            header_lines.append(f"Изменение за 30 дней: {report.price_change_30d:+.1%}")
        if report.sentiment_mean is not None:
            header_lines.append(f"Средний тон новостей: {report.sentiment_mean:+.2f}")
        header_lines.append(f"Количество новостей в анализе: {report.news_count}")
        metrics_block = "\n".join(header_lines)

        summary_text = (report.summary or "").strip() or "Свежая выжимка недоступна."
        summary_block = "Краткая выжимка:\n" + summary_text

        explanation_text = (report.explanation or "").strip()
        explanation_block = (
            "Как интерпретировать:\n" + explanation_text if explanation_text else ""
        )

        highlight_lines: List[str] = []
        for item in report.highlights:
            date = (item.get('date') or '').strip()
            source = (item.get('source') or '').strip()
            title = (item.get('title') or '').strip() or 'Без названия'
            sentiment = item.get('sentiment')
            prefix_parts = [part for part in [date, source] if part]
            prefix = ' | '.join(prefix_parts)
            bullet = f"- {prefix}: {title}" if prefix else f"- {title}"
            try:
                sent_value = float(sentiment)
            except (TypeError, ValueError):
                sent_value = None
            if sent_value is not None:
                try:
                    if not math.isnan(sent_value):
                        bullet += f" (тон {sent_value:+.2f})"
                except TypeError:
                    pass
            url = (item.get('url') or '').strip()
            if url:
                bullet += f"\n  {url}"
            highlight_lines.append(bullet)
        highlight_block = ""
        if highlight_lines:
            highlight_block = "Свежие заметки:\n" + "\n".join(highlight_lines)

        messages = [metrics_block, summary_block]
        if explanation_block:
            messages.append(explanation_block)
        if highlight_block:
            messages.append(highlight_block)
        return [message for message in messages if message.strip()]


class TelegramForecastBot:
    def __init__(self, service: ForecastService) -> None:
        self.service = service

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = (
            "Привет! Я бот RichBich. Команды:\n"
            "- /predict <тикер> — вероятность роста на следующий день\n"
            "- /news <тикер> [дней] — последние новости за указанный период\n"
            "Можно также просто написать название компании, и я помогу найти её сам."
        )
        await self._reply(update, message)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tickers = ", ".join(self.service.list_tickers()) or "нет настроенных тикеров"
        message = (
            "Команды:\n"
            "- /predict <тикер> — вероятность роста на следующий день\n"
            "- /news <тикер> [дней] — последние новости (по умолчанию 3 дня)\n"
            "- свободный текст — я попробую распознать компанию\n"
            f"Доступные тикеры из конфигурации: {tickers}"
        )
        await self._reply(update, message)

    def _extract_args(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> list[str]:
        args = list(context.args) if context.args else []
        if args:
            return args
        message = update.effective_message
        if message and message.text:
            parts = message.text.strip().split()
            if parts:
                return parts[1:]
        return []

    async def predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = self._extract_args(update, context)
        if not args:
            await self._reply(update, "Укажи тикер: /predict AAPL")
            return
        ticker = args[0]
        try:
            result = self.service.latest_prediction(ticker)
        except ValueError as exc:
            await self._reply(update, str(exc))
            return
        message = (
            f"Тикер: {result.ticker}\n"
            f"Дата признаков: {result.date}\n"
            f"Цена закрытия: {result.close:.2f}\n"
            f"Вероятность роста: {result.probability_up:.2%}\n"
            f"Прогноз направления: {result.direction.upper()}"
        )
        await self._reply(update, message)

    async def news(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = self._extract_args(update, context)
        if not args:
            await self._reply(update, "Укажи тикер: /news MSFT [дней]")
            return
        ticker = args[0]
        days = 3
        if len(args) > 1:
            try:
                days = max(1, int(args[1]))
            except ValueError:
                await self._reply(update, "Второй аргумент должен быть числом дней.")
                return
        try:
            items = self.service.recent_news(ticker, days=days)
        except ValueError as exc:
            await self._reply(update, str(exc))
            return
        if not items:
            await self._reply(update, "Свежих новостей не найдено. Попробуй позже или уточни запрос.")
            return
        lines = [f"Последние новости за {days} дн. для {ticker.upper()}:"]
        for item in items:
            title = item["title"] or "(без названия)"
            url = item["url"] or ""
            sentiment = item["sentiment"]
            source = item["source"] or ""
            prefix = f"{item['date']} | {source}" if source else item["date"]
            snippet = f"- {prefix}: {title} (sentiment {sentiment:+.2f})"
            if url:
                snippet += f"\n  {url}"
            lines.append(snippet)
        await self._reply(update, "\n".join(lines))

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text:
            return
        query = message.text.strip()
        pending_matches: List[CompanyMatch] | None = context.user_data.get("pending_matches")
        if pending_matches:
            chosen = self._select_match(query, pending_matches)
            if not chosen:
                await self._reply(update, "Не понял выбор. Напиши номер или тикер из списка.")
                return
            context.user_data.pop("pending_matches", None)
            original_query = context.user_data.pop("pending_query", query)
            await self._send_report(update, chosen, original_query)
            return

        matches = self.service.match_companies(query)
        if not matches:
            await self._reply(update, "Не удалось распознать компанию по запросу. Уточни название или тикер.")
            return
        if len(matches) > 1:
            context.user_data["pending_matches"] = matches
            context.user_data["pending_query"] = query
            await self._reply(update, self._format_match_options(matches))
            return
        await self._send_report(update, matches[0], query)

    async def _send_report(self, update: Update, match: CompanyMatch, query: str) -> None:
        try:
            report = self.service.build_report_from_match(match, original_query=query)
        except ValueError as exc:
            await self._reply(update, str(exc))
            return
        for chunk in self.service.build_messages(report):
            await self._reply(update, chunk)

    @staticmethod
    def _format_match_options(matches: List[CompanyMatch]) -> str:
        lines = ["Нашёл такие компании:"]
        for idx, match in enumerate(matches, 1):
            name = match.shortname or match.longname or match.symbol
            exchange = f" ({match.exchange})" if match.exchange else ""
            lines.append(f"{idx}. {name} [{match.symbol}]{exchange}")
        lines.append("Ответь номером из списка или тикером.")
        return "\n".join(lines)

    @staticmethod
    def _select_match(selection: str, matches: List[CompanyMatch]) -> Optional[CompanyMatch]:
        selection = selection.strip()
        if not selection:
            return None
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
            return None
        upper = selection.replace(" ", "").upper()
        for match in matches:
            if match.symbol.upper() == upper:
                return match
        return None

    async def _reply(self, update: Update, message: str) -> None:
        msg = update.effective_message
        if not msg:
            return
        await msg.reply_text(message)


def build_application(
    token: str,
    cfg: TrainingConfig,
    model_path: Optional[str] = None,
) -> Application:
    service = ForecastService(cfg, model_path=Path(model_path) if model_path else None)
    bot = TelegramForecastBot(service)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("help", bot.help))
    app.add_handler(CommandHandler("predict", bot.predict))
    app.add_handler(CommandHandler("news", bot.news))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    return app


def run_bot(
    token: Optional[str] = None,
    config_path: str | Path = "configs/training.yaml",
    model_path: Optional[str] = None,
) -> None:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Telegram token not provided. Set TELEGRAM_BOT_TOKEN or pass --token.")
    cfg = load_config(config_path)
    application = build_application(token, cfg, model_path=model_path)
    LOGGER.info("Starting Telegram bot...")
    application.run_polling()

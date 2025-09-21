"""News summarisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore


@dataclass
class SummaryResult:
    text: str
    used_model: Optional[str]


class NewsSummarizer:
    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        max_input_chars: int = 6000,
    ) -> None:
        self._max_input_chars = max_input_chars
        self._model_name = model_name
        self._pipeline = None
        if pipeline is None:
            LOGGER.warning("transformers is not available; falling back to heuristic summaries")
            return
        try:
            self._pipeline = pipeline("summarization", model=model_name, tokenizer=model_name)
        except Exception as exc:  # pragma: no cover - optional path
            LOGGER.error("Failed to load summarisation model %s: %s", model_name, exc)
            self._pipeline = None

    def summarise(self, texts: Iterable[str], *, default: str = "Нет достаточных новостей для вывода.") -> SummaryResult:
        joined = " ".join(t.strip() for t in texts if t and t.strip())
        if not joined:
            return SummaryResult(text=default, used_model=None)
        truncated = joined[: self._max_input_chars]
        if self._pipeline is None:
            summary = self._heuristic_summary(truncated)
            return SummaryResult(text=summary, used_model=None)
        try:
            output = self._pipeline(
                truncated,
                max_length=220,
                min_length=60,
                do_sample=False,
                truncation=True,
            )
            if output and isinstance(output, list) and output[0].get("summary_text"):
                return SummaryResult(text=output[0]["summary_text"], used_model=self._model_name)
        except Exception as exc:  # pragma: no cover - optional path
            LOGGER.error("Summarisation failed: %s", exc)
        summary = self._heuristic_summary(truncated)
        return SummaryResult(text=summary, used_model=None)

    @staticmethod
    def _heuristic_summary(text: str, sentences: int = 3) -> str:
        parts = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
        if not parts:
            return "Нет достаточной информации для краткого описания."
        selected = parts[:sentences]
        return ". ".join(selected) + ("." if selected else "")

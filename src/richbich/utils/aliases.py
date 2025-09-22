"""Alias utilities for tickers, queries, and fallbacks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml

from richbich.utils.io import ensure_dir
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALIAS_STORE_PATH = _PROJECT_ROOT / "data" / "metadata" / "company_aliases.yaml"

# Static fallbacks for price retrieval (primary -> list of alternatives)
_PRICE_FALLBACKS: dict[str, list[str]] = {
    "^NDX": ["QQQ", "NDX"],
    "NASDAQ": ["^NDX", "QQQ"],
    "NASDAQ100": ["^NDX", "QQQ"],
    "NAS100": ["^NDX", "QQQ"],
    "^IXIC": ["QQQ"],
    "SP500": ["^GSPC", "SPY"],
    "S&P500": ["^GSPC", "SPY"],
    "^DJI": ["DIA"],
}

# Seed query fallbacks for common tickers
_QUERY_FALLBACKS: dict[str, list[str]] = {
    "AAPL": ["Apple", "iPhone", "苹果"],
    "MSFT": ["Microsoft", "Azure", "微软"],
    "TSLA": ["Tesla", "Илон Маск", "特斯拉"],
    "NVDA": ["NVIDIA", "Nvidia", "英伟达"],
    "BABA": ["Alibaba", "Ali Baba", "阿里巴巴"],
    "^NDX": ["Nasdaq 100", "Nasdaq index", "NDX"],
    "NASDAQ": ["Nasdaq 100", "Nasdaq", "NDX"],
}

_ALIAS_CACHE: dict[str, set[str]] | None = None


def _load_alias_cache() -> dict[str, set[str]]:
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    data: dict[str, list[str]] = {}
    if _ALIAS_STORE_PATH.exists():
        try:
            raw = yaml.safe_load(_ALIAS_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = {str(key): list(val or []) for key, val in raw.items()}
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load alias store %s: %s", _ALIAS_STORE_PATH, exc)
    _ALIAS_CACHE = {
        symbol.upper(): {str(name).strip() for name in names if str(name).strip()}
        for symbol, names in data.items()
    }
    return _ALIAS_CACHE


def _persist_alias_cache() -> None:
    cache = _load_alias_cache()
    ensure_dir(_ALIAS_STORE_PATH.parent)
    serialisable = {symbol: sorted(values) for symbol, values in cache.items() if values}
    try:
        _ALIAS_STORE_PATH.write_text(
            yaml.safe_dump(serialisable, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to persist alias cache to %s: %s", _ALIAS_STORE_PATH, exc)


def register_aliases(symbol: str, names: Iterable[str]) -> bool:
    """Register additional alias strings for a ticker.

    Returns True if the cache was modified.
    """
    cache = _load_alias_cache()
    upper = symbol.strip().upper()
    if not upper:
        return False
    bucket = cache.setdefault(upper, set())
    updated = False
    for name in names:
        if not name:
            continue
        cleaned = str(name).strip()
        if not cleaned:
            continue
        if not any(ch.isalpha() for ch in cleaned):
            # allow digits when length > 1 (e.g., "3M"), otherwise skip
            if len(cleaned) <= 1:
                continue
        if cleaned not in bucket:
            bucket.add(cleaned)
            updated = True
    if updated:
        _persist_alias_cache()
    return updated


def alias_terms_for(symbol: str) -> List[str]:
    cache = _load_alias_cache()
    upper = symbol.strip().upper()
    combined: list[str] = []
    for item in cache.get(upper, set()):
        if item not in combined:
            combined.append(item)
    for seed in _QUERY_FALLBACKS.get(upper, []):
        if seed not in combined:
            combined.append(seed)
    return combined


def query_fallback_terms(symbol: str, base_terms: Iterable[str] | None = None) -> List[str]:
    terms: list[str] = []
    for term in base_terms or []:
        if term and term not in terms:
            terms.append(term)
    for alias in alias_terms_for(symbol):
        if alias and alias not in terms:
            terms.append(alias)
    if not terms:
        terms.append(symbol.upper())
    return terms


def price_fallbacks_for(symbol: str) -> List[str]:
    upper = symbol.strip().upper()
    fallbacks = [upper]
    for candidate in _PRICE_FALLBACKS.get(upper, []):
        candidate_upper = candidate.upper()
        if candidate_upper not in fallbacks:
            fallbacks.append(candidate_upper)
    return fallbacks

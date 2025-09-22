
"""Company lookup utilities combining local registry and remote search."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import List, Sequence

import re
import time
import requests

from richbich.services import company_registry
from richbich.utils.aliases import register_aliases
from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_MAX_REMOTE_RESULTS = 20
_REMOTE_MAX_RETRIES = 3
_REMOTE_RETRY_BACKOFF = 0.6
_CANDIDATE_SYMBOL_RE = re.compile(r"[A-Za-z0-9]{1,10}(?:\.[A-Za-z0-9]{1,3})?")

_FALLBACK_SYMBOLS = {
    "TESLA": ("TSLA", "Tesla Inc."),
    "TSLA": ("TSLA", "Tesla Inc."),
    "NVIDIA": ("NVDA", "NVIDIA Corporation"),
    "NVDA": ("NVDA", "NVIDIA Corporation"),
    "ALIBABA": ("BABA", "Alibaba Group Holding Ltd"),
    "BABA": ("BABA", "Alibaba Group Holding Ltd"),
    "MICROSOFT": ("MSFT", "Microsoft Corporation"),
    "MSFT": ("MSFT", "Microsoft Corporation"),
    "APPLE": ("AAPL", "Apple Inc."),
    "AAPL": ("AAPL", "Apple Inc."),
    "AMAZON": ("AMZN", "Amazon.com Inc."),
    "AMZN": ("AMZN", "Amazon.com Inc."),
    "GOOGLE": ("GOOGL", "Alphabet Inc."),
    "ALPHABET": ("GOOGL", "Alphabet Inc."),
    "META": ("META", "Meta Platforms Inc."),
    "FACEBOOK": ("META", "Meta Platforms Inc."),
    "YANDEX": ("YNDX", "Yandex N.V."),
    "GAZPROM": ("GAZP.ME", "Gazprom PJSC"),
    "SBERBANK": ("SBER.ME", "Sberbank"),
    "TESLAINC": ("TSLA", "Tesla Inc."),
    "NASDAQ": ("^NDX", "Nasdaq 100 Index"),
    "NASDAQ100": ("^NDX", "Nasdaq 100 Index"),
    "NAS100": ("^NDX", "Nasdaq 100 Index"),
}

@dataclass
class CompanyMatch:
    symbol: str
    shortname: str | None
    longname: str | None
    exchange: str | None
    score: float | None


def _normalise_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _extract_candidate_symbols(text: str) -> List[str]:
    cleaned = re.sub(r"[^0-9A-Za-z\. ]+", " ", (text or ""))
    tokens: set[str] = set()
    for token in cleaned.replace('-', ' ').split():
        if not token:
            continue
        match = _CANDIDATE_SYMBOL_RE.fullmatch(token)
        if match:
            tokens.add(match.group().upper())
            continue
        if any(ch.isalpha() for ch in token) and len(token) <= 12:
            tokens.add(token.upper())
    ordered = sorted(tokens, key=lambda item: (len(item), item))
    return ordered


@lru_cache(maxsize=256)
def _remote_lookup_cached(query: str) -> Sequence[CompanyMatch]:
    return tuple(_remote_lookup_uncached(query, _MAX_REMOTE_RESULTS))


def _remote_lookup(query: str, limit: int) -> List[CompanyMatch]:
    if not query.strip():
        return []
    cached = list(_remote_lookup_cached(query.strip()))
    return cached[:limit]


def _remote_lookup_uncached(query: str, limit: int) -> List[CompanyMatch]:
    if not query.strip():
        return []
    params = {
        "q": query,
        "quotesCount": max(limit, 5),
        "newsCount": 0,
        "lang": "en-US",
    }
    payload: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, _REMOTE_MAX_RETRIES + 1):
        try:
            response = requests.get(YAHOO_SEARCH_URL, params=params, timeout=6)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.HTTPError as exc:  # pragma: no cover - network issues
            status = exc.response.status_code if exc.response is not None else None
            LOGGER.warning("Remote company lookup failed for '%s' (status %s): %s", query, status, exc)
            last_error = exc
            if status == 429 and attempt < _REMOTE_MAX_RETRIES:
                time.sleep(_REMOTE_RETRY_BACKOFF * attempt)
                continue
            return []
        except requests.RequestException as exc:  # pragma: no cover - network issues
            LOGGER.warning("Remote company lookup failed for '%s': %s", query, exc)
            last_error = exc
        except ValueError as exc:  # pragma: no cover - unexpected payload
            LOGGER.warning("Remote company lookup returned non-JSON payload for '%s'", query)
            last_error = exc
        if attempt < _REMOTE_MAX_RETRIES:
            time.sleep(_REMOTE_RETRY_BACKOFF * attempt)
    else:  # pragma: no cover - defensive fallback
        if last_error is not None:
            LOGGER.warning("Remote lookup exhausted retries for '%s': %s", query, last_error)
        return []

    quotes = (payload.get("quotes") if isinstance(payload, dict) else None) or []
    matches: List[CompanyMatch] = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol:
            continue
        shortname = quote.get("shortname") or quote.get("displayName")
        longname = quote.get("longname") or quote.get("companyName") or shortname
        exchange = quote.get("exchDisp") or quote.get("exchange")
        score = quote.get("score")
        if isinstance(score, (int, float)):
            score_value: float | None = float(score)
        else:
            score_value = None
        matches.append(
            CompanyMatch(
                symbol=_normalise_symbol(symbol),
                shortname=shortname,
                longname=longname,
                exchange=exchange,
                score=score_value,
            )
        )
        if len(matches) >= limit:
            break
    return matches


def search_company(query: str, *, count: int = 5) -> List[CompanyMatch]:
    records = list(company_registry.iter_records())
    if not query or not query.strip():
        return []

    query = query.strip()
    norm_query = company_registry.normalise_query(query)
    norm_tokens = {norm_query}
    norm_tokens.update(
        company_registry.normalise_query(part)
        for part in query.replace('/', ' ').split()
        if part.strip()
    )

    matches: List[CompanyMatch] = []
    seen_symbols: set[str] = set()

    def _register(symbol: str, *aliases: str) -> None:
        register_aliases(symbol, [alias for alias in aliases if isinstance(alias, str) and alias.strip()])

    def _push(match: CompanyMatch, *, aliases: Sequence[str] | None = None) -> None:
        symbol = _normalise_symbol(match.symbol)
        if symbol in seen_symbols:
            return
        seen_symbols.add(symbol)
        matches.append(match)
        if aliases:
            _register(symbol, *aliases)

    def _push_record(record, score: float, aliases: Sequence[str] | None = None) -> None:
        _push(
            CompanyMatch(
                symbol=_normalise_symbol(record.symbol),
                shortname=record.company,
                longname=record.company,
                exchange=record.exchange,
                score=score,
            ),
            aliases=aliases,
        )

    exact = company_registry.find_by_symbol(query)
    if exact:
        _push_record(exact, 1.0, aliases=(query, exact.company))

    if len(matches) < count:
        for record in records:
            if record.symbol in seen_symbols:
                continue
            if any(token in record.aliases for token in norm_tokens):
                _push_record(record, 0.9, aliases=(query, record.company))
                if len(matches) >= count:
                    break

    if len(matches) < count:
        alias_map = {alias: record for record in records for alias in record.aliases}
        for token in norm_tokens:
            for alias in get_close_matches(token, alias_map.keys(), n=count, cutoff=0.6):
                record = alias_map[alias]
                if record.symbol in seen_symbols:
                    continue
                _push_record(record, 0.75, aliases=(query, record.company))
                if len(matches) >= count:
                    break
            if len(matches) >= count:
                break

    if len(matches) < count:
        for candidate in _extract_candidate_symbols(query):
            normalized_candidate = _normalise_symbol(candidate)
            if normalized_candidate in seen_symbols:
                continue
            fallback = _FALLBACK_SYMBOLS.get(normalized_candidate)
            if fallback:
                fallback_symbol, default_name = fallback
                record = company_registry.find_by_symbol(fallback_symbol)
                if record:
                    _push_record(record, 0.88, aliases=(query, default_name))
                else:
                    _push(
                        CompanyMatch(
                            symbol=_normalise_symbol(fallback_symbol),
                            shortname=default_name,
                            longname=default_name,
                            exchange=None,
                            score=0.6,
                        ),
                        aliases=(query, default_name),
                    )
                if len(matches) >= count:
                    break
                continue
            record = company_registry.find_by_symbol(candidate)
            if record:
                _push_record(record, 0.85, aliases=(query, record.company))
            else:
                remote_single = _remote_lookup(candidate, 1)
                if remote_single:
                    remote_match = remote_single[0]
                    _push(remote_match, aliases=(query, remote_match.shortname or "", remote_match.longname or ""))
                else:
                    _push(
                        CompanyMatch(
                            symbol=normalized_candidate,
                            shortname=candidate,
                            longname=candidate,
                            exchange=None,
                            score=None,
                        ),
                        aliases=(query,),
                    )
            if len(matches) >= count:
                break

    if len(matches) < count:
        remote_limit = min(_MAX_REMOTE_RESULTS, max(count - len(matches), count))
        for remote in _remote_lookup(query, remote_limit):
            _push(remote, aliases=(query, remote.shortname or "", remote.longname or ""))
            if len(matches) >= count:
                break

    if not matches:
        compact = query.replace(" ", "")
        upper = _normalise_symbol(compact)
        if compact and compact.isalnum() and 1 <= len(compact) <= 12:
            fallback = _FALLBACK_SYMBOLS.get(upper)
            if fallback:
                symbol, default_name = fallback
                _push(
                    CompanyMatch(_normalise_symbol(symbol), default_name, default_name, None, 0.6),
                    aliases=(query, default_name),
                )
            else:
                _push(CompanyMatch(upper, upper, upper, None, None), aliases=(query,))

    return matches[:count]





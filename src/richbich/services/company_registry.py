"""Shared company registry utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import csv

from richbich.utils.logging import get_logger
from richbich.utils.aliases import alias_terms_for

LOGGER = get_logger(__name__)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path.cwd()
_INDEX_CANDIDATES = [
    _PACKAGE_ROOT / "configs" / "company_index.csv",
    _PROJECT_ROOT / "configs" / "company_index.csv",
    Path(__file__).resolve().parents[3] / "configs" / "company_index.csv",
]
_ALIAS_SPLIT = re.compile(r"[;,]")
_NORMALISE_PATTERN = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")


@dataclass(frozen=True)
class CompanyRecord:
    symbol: str
    company: str
    exchange: str | None
    aliases: Sequence[str]


def _normalise(text: str) -> str:
    cleaned = _NORMALISE_PATTERN.sub(" ", text).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _find_index_path() -> Path | None:
    for candidate in _INDEX_CANDIDATES:
        if candidate.exists():
            return candidate
    LOGGER.warning("Company index not found in default locations (%s)", _INDEX_CANDIDATES)
    return None


@lru_cache(maxsize=1)
def load_company_records() -> List[CompanyRecord]:
    index_path = _find_index_path()
    if index_path is None:
        return []
    records: List[CompanyRecord] = []
    with index_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = { (key.lstrip("\ufeff") if isinstance(key, str) else key): value for key, value in row.items() }
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            company = (row.get("company") or symbol).strip()
            exchange = (row.get("exchange") or "").strip() or None
            aliases_field = row.get("aliases") or ""
            aliases = {
                _normalise(token)
                for token in _ALIAS_SPLIT.split(aliases_field)
                if token.strip()
            }
            aliases.add(_normalise(symbol))
            aliases.add(_normalise(company))
            records.append(
                CompanyRecord(
                    symbol=symbol,
                    company=company,
                    exchange=exchange,
                    aliases=tuple(sorted(alias for alias in aliases if alias)),
                )
            )
    LOGGER.info("Loaded %d company records from %s", len(records), index_path)
    return records


def symbol_aliases(symbol: str) -> Sequence[str]:
    upper = symbol.strip().upper()
    for record in load_company_records():
        if record.symbol == upper:
            dynamic = {
                _normalise(term)
                for term in alias_terms_for(upper)
                if _normalise(term)
            }
            aliases = set(record.aliases) | dynamic
            return tuple(sorted(aliases))
    dynamic_only = {
        _normalise(term)
        for term in alias_terms_for(upper)
        if _normalise(term)
    }
    return tuple(sorted(dynamic_only))





def find_by_symbol(symbol: str) -> CompanyRecord | None:
    upper = symbol.strip().upper()
    for record in load_company_records():
        if record.symbol == upper:
            return record
    return None


def iter_records() -> Iterable[CompanyRecord]:
    return load_company_records()


def normalise_query(text: str) -> str:
    return _normalise(text)


"""Company lookup utilities using local registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from difflib import get_close_matches

from richbich.services import company_registry


@dataclass
class CompanyMatch:
    symbol: str
    shortname: str | None
    longname: str | None
    exchange: str | None
    score: float | None


def search_company(query: str, *, count: int = 5) -> List[CompanyMatch]:
    records = list(company_registry.iter_records())
    if not query or not query.strip():
        return []

    query = query.strip()
    norm_query = company_registry.normalise_query(query)

    matches: List[CompanyMatch] = []
    exact = company_registry.find_by_symbol(query)
    if exact:
        matches.append(
            CompanyMatch(exact.symbol, exact.company, exact.company, exact.exchange, 1.0)
        )

    for record in records:
        if record in matches:
            continue
        if norm_query in record.aliases:
            matches.append(
                CompanyMatch(record.symbol, record.company, record.company, record.exchange, 0.9)
            )

    if len(matches) < count:
        aliases = {alias: record for record in records for alias in record.aliases}
        for alias in get_close_matches(norm_query, aliases.keys(), n=count, cutoff=0.6):
            record = aliases[alias]
            if any(match.symbol == record.symbol for match in matches):
                continue
            matches.append(
                CompanyMatch(record.symbol, record.company, record.company, record.exchange, 0.75)
            )

    if not matches:
        compact = query.replace(" ", "")
        if compact.isalnum() and compact.isupper() and 1 <= len(compact) <= 6:
            matches.append(CompanyMatch(compact, compact, compact, None, None))

    return matches[:count]

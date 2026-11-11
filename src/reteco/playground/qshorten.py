"""Extractive query shortening for the first stage (E7).

A query is a StackExchange post: title on the first line, HTML body after it. The short
query keeps the title (repeated `title_weight` times), the first `first_n` sentences of the
cleaned body, and (if `time_sents`) every other body sentence that contains a time
expression (year, decade, century, month, era marker or a present-time word).
`first_n: null` keeps the whole body (only the title repetition then changes anything).

Config block (top level of a pipeline config):
    query_shorten: {first_n: 2, time_sents: true, title_weight: 1}
"""
from __future__ import annotations

import re

from ..data import Query
from ..sparse.clean import clean_query

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TIME = re.compile(
    r"\b(1[0-9]{3}|20[0-2][0-9])s?\b|\b'\d0s\b|\b\d{1,2}(st|nd|rd|th)[- ]century\b|"
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b(bc|bce|ad|ce)\b|\b(century|centuries|decade|decades|era|medieval|ancient|modern|"
    r"nowadays|currently|today|recently|nowaday|present[- ]day)\b", re.I)


def shorten(text: str, first_n=2, time_sents=True, title_weight=1) -> str:
    title, _, body = text.partition("\n")
    title = clean_query(title)
    body = clean_query(body)
    sents = [s for s in _SENT.split(body) if s.strip()]
    if first_n is None:
        keep = sents
    else:
        keep = [s for i, s in enumerate(sents) if i < first_n or (time_sents and _TIME.search(s))]
    return " ".join([title] * title_weight + keep)


def shorten_queries(qs: list[Query], cfg: dict | None, stage: str | None = None) -> list[Query]:
    """Queries are returned unchanged when cfg is empty or `stage` is not in cfg["stages"]
    (default: both first-stage retrievers). The output is plain text (already cleaned), so
    the stages' own cleaning is a no-op on it."""
    if not cfg or (stage is not None and stage not in cfg.get("stages", ("sparse", "dense"))):
        return qs
    return [Query(q.id, shorten(q.query, cfg.get("first_n", 2), cfg.get("time_sents", True),
                                cfg.get("title_weight", 1))) for q in qs]

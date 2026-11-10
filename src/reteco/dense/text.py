"""Text preparation: HTML stripping for queries, token-window passages for documents."""
from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_query(text: str) -> str:
    """Queries are StackExchange posts (title + HTML body): drop tags, unescape, squash spaces."""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def split_passages(texts: list[str], tokenizer, window: int, stride: int) -> tuple[list[str], list[int]]:
    """Cut each text into windows of `window` tokens starting every `stride` tokens.

    Windows are cut on the model tokenizer's character offsets, so each passage is
    a verbatim substring of the original text. Returns (passages, owner) where
    owner[i] is the index in `texts` of passage i. Every text yields >= 1 passage
    and a text's passages are contiguous and in order.
    """
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=False, verbose=False)
    passages, owner = [], []
    for i, (text, offsets) in enumerate(zip(texts, enc["offset_mapping"])):
        n = len(offsets)
        if n <= window:
            passages.append(text)
            owner.append(i)
            continue
        start = 0
        while True:
            end = min(start + window, n)
            passages.append(text[offsets[start][0]:offsets[end - 1][1]])
            owner.append(i)
            if end == n:
                break
            start += stride
    return passages, owner

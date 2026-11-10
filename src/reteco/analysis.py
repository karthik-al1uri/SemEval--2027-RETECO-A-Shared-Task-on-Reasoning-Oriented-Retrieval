"""Pure-Python approximation of Lucene's EnglishAnalyzer (pyserini's default analyzer).

Lucene chain: StandardTokenizer -> EnglishPossessiveFilter -> LowerCaseFilter
-> StopFilter (English, 33 words) -> PorterStemFilter.

Here:
  * StandardTokenizer (Unicode UAX#29 word breaks) is approximated by a regex:
    runs of word characters, joined across `. : ' ’ ·` between letters and
    across `, . ; ' ’` between digits ("U.S.", "3.14", "1,000", "don't" stay
    whole; hyphens, slashes and other punctuation split).
  * Tokens made only of underscores are dropped (Lucene only emits tokens that
    contain a letter or digit).
  * Porter stemming uses NLTK's PorterStemmer in MARTIN_EXTENSIONS mode, i.e.
    Martin Porter's own reference implementation that Lucene's PorterStemmer
    is ported from (bli->ble, logi->log, words of <= 2 letters unchanged).
    The original-paper variant (e.g. PyStemmer "porter") gives slightly
    different stems and measurably different BM25 scores.

Known differences from Lucene: CJK ideographs are not split into single
characters, emoji are dropped, and tokens longer than 255 characters are not
split. These are rare in the TEMPO corpora.
"""
from __future__ import annotations

import re

from nltk.stem.porter import PorterStemmer

ANALYZER_VERSION = "lucene_english_v2"

# Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the
their then there these they this to was will with
""".split())

_LETTER = r"[^\W\d_]"
TOKEN_RE = re.compile(
    rf"\w+(?:(?:(?<={_LETTER})[.:'’·](?={_LETTER})|(?<=\d)[,.;'’](?=\d))\w+)*"
)
_POSSESSIVE = ("'s", "'S", "’s", "’S", "＇s", "＇S")


class LuceneEnglishAnalyzer:
    def __init__(self):
        self._stem = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS).stem
        self._cache: dict[str, str | None] = {}

    def _norm(self, tok: str) -> str | None:
        out = self._cache.get(tok, False)
        if out is False:
            t = tok[:-2] if tok.endswith(_POSSESSIVE) else tok
            t = t.lower()
            if not t.strip("_") or t in STOPWORDS:
                out = None
            else:
                out = self._stem(t, to_lowercase=False)
            if len(self._cache) < 5_000_000:
                self._cache[tok] = out
        return out

    def analyze(self, text: str) -> list[str]:
        norm = self._norm
        return [t for t in map(norm, TOKEN_RE.findall(text)) if t is not None]

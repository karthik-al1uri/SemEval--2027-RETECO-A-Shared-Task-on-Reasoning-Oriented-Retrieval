import math

import pytest

from reteco.analysis import LuceneEnglishAnalyzer
from reteco.bm25 import BM25Index


def test_analyzer_matches_lucene_english_behaviour():
    a = LuceneEnglishAnalyzer()
    assert a.analyze("The U.S. economy's growth in 2019-2020 was 3.14% (1,000 units)") == [
        "u.", "economi", "growth", "2019", "2020", "3.14", "1,000", "unit"]
    # Martin Porter variant (as in Lucene), not the original paper's rules.
    assert a.analyze("possibly technological") == ["possibl", "technolog"]
    # Stopwords removed; underscore-only tokens dropped; underscores inside words kept.
    assert a.analyze("this is a ___ foo_bar") == ["foo_bar"]


def test_bm25_matches_gensim_lucene_formula(tiny_data):
    index = BM25Index.get("iota", k1=0.9, b=0.4, workers=1, root=tiny_data)
    a = LuceneEnglishAnalyzer()
    docs = {"iota/d1": "IOTA Chrysalis migration in 2021.",
            "iota/d2": "The legacy network was shut down.",
            "iota/d3": "Unrelated text about tangles."}
    toks = {d: a.analyze(t) for d, t in docs.items()}
    n, avgdl = len(docs), sum(map(len, toks.values())) / len(docs)

    def w(term, bag):
        df = sum(term in t for t in toks.values())
        idf = math.log(n + 1) - math.log(df + 0.5)
        tf = bag.count(term)
        return idf * tf / (tf + 0.9 * (1 - 0.4 + 0.4 * len(bag) / avgdl))

    query = a.analyze("chrysalis migration migration network")
    got = index.search(query, depth=3)
    q_in_vocab = [t for t in query if t in index.vocab]
    for d, bag in toks.items():
        expected = sum(w(t, q_in_vocab) * w(t, bag) for t in set(q_in_vocab))
        assert got[d] == pytest.approx(expected, rel=1e-5)
    assert max(got, key=got.get) == "iota/d1"


def test_oov_query_returns_zero_scores(tiny_data):
    index = BM25Index.get("quant", k1=0.9, b=0.4, workers=1, root=tiny_data)
    got = index.search(["zzzz"], depth=10)
    assert len(got) == 2 and set(got.values()) == {0.0}

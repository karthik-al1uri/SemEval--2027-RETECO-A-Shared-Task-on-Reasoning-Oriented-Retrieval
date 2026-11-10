import numpy as np

from reteco.analysis_tools.errors import doc_prox, ranks, stratified_sample
from reteco.analysis_tools.perquery import macro, per_query
from reteco.analysis_tools.qtypes import INTENT_MAP, TYPES, query_facets, rule_type, type_from_guidance
from reteco.analysis_tools.significance import (bootstrap_ci, holm, macro_diff,
                                                randomization_test)


def test_macro_diff_weights_domains_equally():
    a = [1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    doms = ["x", "x", "x", "y"]
    assert macro_diff(a, b, doms) == 0.5          # (1 + 0) / 2, not 3/4


def test_randomization_identical_and_clear_difference():
    doms = ["x"] * 30 + ["y"] * 30
    a = np.full(60, 0.5)
    assert randomization_test(a, a, doms, n=200) == 1.0
    assert randomization_test(a + 0.3, a, doms, n=500) < 0.01


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(0)
    doms = ["x"] * 50 + ["y"] * 50
    a = rng.random(100) + 0.1
    b = a - 0.1
    lo, hi = bootstrap_ci(a, b, doms, n=500)
    assert lo <= 0.1 + 1e-9 and hi >= 0.1 - 1e-9


def test_holm_is_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == 0.04 and max(adj) <= 1.0
    assert adj[2] == adj[1] == 0.09


def test_stratified_sample_covers_domains():
    pool = {"a": [f"a{i}" for i in range(20)], "b": ["b0"], "c": [f"c{i}" for i in range(5)]}
    s = stratified_sample(pool, 6)
    assert len(s) == len(set(s)) == 6
    assert {x[0] for x in s} == {"a", "b", "c"}
    assert s == stratified_sample(pool, 6)          # deterministic


def test_ranks_follow_trec_order():
    r = ranks({"q": {"d1": 1.0, "d2": 2.0, "d3": 1.0}})["q"]
    assert r == {"d2": 1, "d3": 2, "d1": 3}


def test_doc_prox():
    assert doc_prox([(1940, 1945)], [(1942, 1942)]) == 1.0
    assert doc_prox([], [(1942, 1942)]) is None
    assert doc_prox([(1940, 1945)], [(1942, 1942), (2020, 2020)]) < 0.6


def test_type_mapping_covers_types():
    assert set(INTENT_MAP.values()) == set(TYPES)
    assert type_from_guidance({"is_temporal_query": False, "temporal_intent": "when"}) == "no_time"
    assert type_from_guidance({"is_temporal_query": True, "temporal_intent": "duration"}) == "duration"


def test_rules_on_simple_queries():
    assert rule_type(query_facets("How long does a block take to confirm?\n<p>x</p>")) == "duration"
    assert rule_type(query_facets("Is Latin still taught?\n<p>x</p>")) == "present_vs_past"
    assert rule_type(query_facets("When did the Ming dynasty end?\n<p>x</p>")) == "specific_date"


def test_per_query_matches_macro(tiny_data):
    run = {"q3": {"iota/d3": 1.0}, "q5": {"quant/d1": 1.0}}
    rows = per_query(run, "dev", domains=["iota", "quant"], root=tiny_data)
    assert [r["ndcg@10"] for r in rows] == [1.0, 0.0]
    assert macro(rows, "ndcg@10") == 0.5

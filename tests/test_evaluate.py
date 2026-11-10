import csv
import math

import pytest

from reteco import evaluate
from reteco.runs import write_run


def ndcg(rels, n_gold, k=10):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / idcg


def test_perfect_and_partial_scores(tiny_data):
    run = {
        "q1": {"iota/d1": 3.0, "iota/d2": 2.0},  # perfect
        "q2": {"iota/d3": 3.0, "iota/d2": 2.0},  # 1 of 2 gold, at rank 2
        "q4": {"quant/d2": 2.0, "quant/d1": 1.0},  # gold at rank 2
    }
    res = evaluate.evaluate_run(run, "train", domains=["iota", "quant"], root=tiny_data)
    iota = res["domains"]["iota"]
    assert iota["ndcg@10"] == pytest.approx((1.0 + ndcg([0, 1], 2)) / 2)
    assert iota["recall@100"] == pytest.approx((1.0 + 0.5) / 2)
    assert res["domains"]["quant"]["ndcg@10"] == pytest.approx(ndcg([0, 1], 1))
    assert res["macro_ndcg@10"] == pytest.approx(
        (iota["ndcg@10"] + res["domains"]["quant"]["ndcg@10"]) / 2)


def test_missing_query_scores_zero(tiny_data):
    run = {"q1": {"iota/d1": 1.0}}  # q2 missing
    res = evaluate.evaluate_run(run, "train", domains=["iota"], root=tiny_data)
    assert res["domains"]["iota"]["ndcg@10"] == pytest.approx(0.5)
    assert res["domains"]["iota"]["missing_queries"] == 1


def test_ranking_uses_scores_not_input_order(tiny_data):
    run = {"q4": {"quant/d2": 1.0, "quant/d1": 5.0}}
    res = evaluate.evaluate_run(run, "train", domains=["quant"], root=tiny_data)
    assert res["domains"]["quant"]["ndcg@10"] == pytest.approx(1.0)


def test_only_top10_counts_for_ndcg(tiny_data):
    run = {"q3": {f"iota/x{i}": 100.0 - i for i in range(10)} | {"iota/d3": 1.0}}
    res = evaluate.evaluate_run(run, "dev", domains=["iota"], root=tiny_data)
    assert res["domains"]["iota"]["ndcg@10"] == 0.0
    assert res["domains"]["iota"]["recall@100"] == 1.0


def test_default_domains_and_unknown_queries(tiny_data):
    run = {"q3": {"iota/d3": 1.0}, "zzz": {"iota/d1": 1.0}}
    res = evaluate.evaluate_run(run, "dev", domains=["iota", "quant"], root=tiny_data)
    assert res["unknown_queries"] == 1
    assert res["domains"]["quant"]["ndcg@10"] == 0.0  # requested but absent -> 0


def test_cli_appends_csv(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setenv("RETECO_DATA", str(tiny_data))
    csv_path = tmp_path / "results" / "experiments.csv"
    monkeypatch.setattr(evaluate, "RESULTS_CSV", csv_path)
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    run_path = write_run({"q3": {"iota/d3": 1.0}, "q5": {"quant/d2": 1.0}},
                         tmp_path / "runs" / "dev" / "toy.trec", tag="toy")
    run_path.with_suffix(".meta.json").write_text('{"description": "toy", "config": "configs/toy.yaml"}')
    for _ in range(2):
        assert evaluate.main([str(run_path), "--split", "dev", "--domains", "iota", "quant"]) == 0
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2
    assert rows[0]["system_name"] == "toy"
    assert rows[0]["macro_ndcg@10"] == "1.0000"
    assert rows[0]["iota"] == "1.0000" and rows[0]["bitcoin"] == ""
    assert rows[0]["description"] == "toy" and rows[0]["config"] == "configs/toy.yaml"

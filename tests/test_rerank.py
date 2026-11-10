import copy

import numpy as np
import pytest

from reteco.rerank import interpolate as I


CANDS = {"q1": [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)]}
CE = {"q1": {"a": -2.0, "b": 3.0, "c": 1.0, "d": 0.0, "e": 9.0}}


def order(run, q="q1"):
    return [d for d, _ in sorted(run[q].items(), key=lambda x: -x[1])]


def test_norm_constant_is_zero():
    for m in I.NORMS:
        assert np.all(I.norm(np.array([2.0, 2.0, 2.0]), m) == 0)


def test_norm_ranges():
    x = np.array([1.0, 2.0, 4.0])
    assert I.norm(x, "minmax").tolist() == [0.0, 1 / 3, 1.0]
    z = I.norm(x, "zscore")
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    with pytest.raises(ValueError):
        I.norm(x, "rank")


def test_a0_is_reranker_order_within_depth_and_first_stage_below():
    run = I.combine(CANDS, CE, depth=4, a=0.0)
    # top 4 reranked by CE (e is below the depth, so its CE score is ignored)
    assert order(run) == ["b", "c", "d", "a", "e"]


def test_a1_is_first_stage_order():
    run = I.combine(CANDS, CE, depth=5, a=1.0, method="zscore")
    assert order(run) == ["a", "b", "c", "d", "e"]


def test_out_depth_keeps_best():
    run = I.combine(CANDS, CE, depth=5, a=0.0, out_depth=2)
    assert order(run) == ["e", "b"]


def test_interpolation_midpoint():
    # minmax: fs -> [1, .75, .5, .25, 0]; ce -> [0, 5/11, 3/11, 2/11, 1]
    run = I.combine(CANDS, CE, depth=5, a=0.5)
    fs = np.array([1, .75, .5, .25, 0])
    ce = np.array([0, 5, 3, 2, 11]) / 11
    for (d, _), want in zip(CANDS["q1"], 0.5 * fs + 0.5 * ce):
        assert run["q1"][d] == pytest.approx(want)


def test_score_cache_roundtrip_and_latency_shares(tmp_path, monkeypatch):
    from reteco.rerank import cross_encoder as X

    monkeypatch.setattr(X, "CACHE_DIR", tmp_path)
    cfg = {"model": "org/model", "query_clean": ["html"]}
    c = X.ScoreCache(cfg, "train")
    c.add("iota", "q1", ["iota/a", "iota/b"], np.array([1.5, -0.5]), 0.2)
    c.add("iota", "q1", ["iota/c"], np.array([0.25]), 0.3)
    fresh = X.ScoreCache(cfg, "train")
    fresh.load("iota")
    assert fresh.scores["q1"] == {"iota/a": 1.5, "iota/b": -0.5, "iota/c": 0.25}
    assert fresh.secs["q1"]["iota/a"] == pytest.approx(0.1)
    assert fresh.secs["q1"]["iota/c"] == pytest.approx(0.3)


def test_model_key_changes_with_query_cleaning():
    from reteco.rerank.cross_encoder import model_key

    a = {"model": "BAAI/bge-reranker-base", "query_clean": ["html"]}
    b = dict(a, query_clean=[])
    assert model_key(a) != model_key(b)
    assert model_key(a).startswith("BAAI-bge-reranker-base__")


def test_ablations_each_change_one_thing():
    from reteco.pipeline import ABLATIONS

    base = {"components": {}, "rerank": {"query_clean": ["html"]}}
    seen = set()
    for name, (_, edit) in ABLATIONS.items():
        c = copy.deepcopy(base)
        edit(c)
        assert c != base, name
        key = repr(c)
        assert key not in seen, name
        seen.add(key)


def test_latency_per_stage():
    from reteco.pipeline import Timer, latency

    t = Timer()
    t.add("rerank_dev", 2.0)
    t.add("temporal_dev", 0.5)
    t.add("rerank_train", 9.0)
    lat = latency(t, "dev", 10)
    assert lat == {"rerank": 200.0, "temporal": 50.0, "total": 250.0}


class _FakeTok:
    """Whitespace 'tokenizer' with the two calls prepare_query uses."""

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return {"input_ids": text.split()[:max_length]}

    def decode(self, ids):
        return " ".join(ids)


def test_prepare_query_uses_callers_config_not_encoders():
    # Regression: a reused encoder applied its own query cleaning to another config's queries.
    from reteco.rerank.cross_encoder import CrossEncoder

    enc = CrossEncoder.__new__(CrossEncoder)
    enc.tok = _FakeTok()
    enc.cfg = {"model": "m", "query_clean": ["html"], "max_query_tokens": 256}
    raw = "<p>When did <b>Chrysalis</b> ship?</p>"
    assert "<p>" not in enc.prepare_query(raw)
    assert enc.prepare_query(raw, {"model": "m", "query_clean": []}).startswith("<p>When")
    assert enc.prepare_query("a b c d", {"model": "m", "max_query_tokens": 2}) == "a b"
    assert enc.compatible({"model": "m", "query_clean": []})
    assert not enc.compatible({"model": "other"})


def test_pipeline_dedup_stage_is_optional():
    from reteco.pipeline import describe

    cfg = {"components": {"rerank": False, "temporal": False},
           "fusion": {"k": 20, "w_dense": 0.5}, "dedup": {"config": "configs/final_dedup_end.yaml"}}
    assert describe(cfg).endswith("dedup final_dedup_end")
    cfg["components"]["dedup"] = False
    assert "dedup" not in describe(cfg)
    del cfg["dedup"], cfg["components"]["dedup"]
    assert "dedup" not in describe(cfg)      # configs without a dedup section still work

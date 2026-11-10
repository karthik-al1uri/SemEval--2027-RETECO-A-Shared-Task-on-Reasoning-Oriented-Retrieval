import pytest

from reteco import data


def test_load_queries_exposes_only_id_and_query(tiny_data):
    qs = data.load_queries("iota", "train", root=tiny_data)
    assert [q.id for q in qs] == ["q1", "q2"]
    assert qs[0].query == "When did the Chrysalis migration happen?"
    assert set(vars(qs[0])) == {"id", "query"}  # no gold_ids / gold_answers


def test_iter_corpus_streams_and_skips_blank_lines(tiny_data):
    it = data.iter_corpus("iota", root=tiny_data)
    assert next(it) == ("iota/d1", "IOTA Chrysalis migration in 2021.")
    assert [d for d, _ in it] == ["iota/d2", "iota/d3"]


def test_load_qrels(tiny_data):
    assert data.load_qrels("iota", "train", root=tiny_data) == {
        "q1": {"iota/d1": 1}, "q2": {"iota/d2": 1, "iota/d1": 1}}
    assert data.load_qrels("quant", "dev", root=tiny_data) == {"q5": {"quant/d2": 1}}


def test_unknown_domain_or_split_rejected(tiny_data):
    with pytest.raises(ValueError):
        data.load_queries("nope", "train", root=tiny_data)
    with pytest.raises(ValueError):
        data.load_qrels("iota", "test", root=tiny_data)


def test_data_dir_env_override(monkeypatch, tiny_data):
    monkeypatch.setenv("RETECO_DATA", str(tiny_data))
    assert data.data_dir() == tiny_data
    assert len(data.load_queries("quant", "dev")) == 1


def test_domain_lists():
    assert len(data.DOMAINS) == 13
    assert set(data.SMALL_DOMAINS) <= set(data.DOMAINS)

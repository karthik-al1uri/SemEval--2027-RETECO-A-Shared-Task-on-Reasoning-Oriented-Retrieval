import pytest

from reteco.dense.fusion import ranks, rrf
from reteco.runs import read_run, write_run

K = 60


def test_ranks_follow_trec_eval_order():
    # score descending, ties broken by docid descending
    got = ranks({"q": {"a": 1.0, "b": 1.0, "c": 2.0}})
    assert got == {"q": {"c": 1, "b": 2, "a": 3}}


def test_missing_documents_contribute_zero():
    sparse = {"q": {"a": 1, "b": 2}}
    dense = {"q": {"b": 1, "c": 2}}
    got = rrf([sparse, dense], [0.4, 0.6], K)["q"]
    assert got["a"] == pytest.approx(0.4 / (K + 1))            # absent from dense
    assert got["c"] == pytest.approx(0.6 / (K + 2))            # absent from sparse
    assert got["b"] == pytest.approx(0.4 / (K + 2) + 0.6 / (K + 1))
    assert list(got) == ["b", "c", "a"]


def test_query_missing_from_one_list_uses_the_other():
    sparse = {"q1": {"a": 1, "b": 2}, "q2": {"x": 1}}
    dense = {"q1": {"b": 1}}
    got = rrf([sparse, dense], [0.5, 0.5], K)
    assert got["q2"] == {"x": pytest.approx(0.5 / (K + 1))}


def test_zero_weight_reproduces_the_other_list():
    sparse = {"q": {"a": 1, "b": 2, "c": 3}}
    dense = {"q": {"c": 1, "b": 2, "d": 3}}
    assert list(rrf([sparse, dense], [0.0, 1.0], K)["q"]) == ["c", "b", "d"]
    assert list(rrf([sparse, dense], [1.0, 0.0], K)["q"]) == ["a", "b", "c"]


def test_weights_normalise():
    sparse = {"q": {"a": 1, "b": 2, "c": 3, "d": 4}}
    dense = {"q": {"d": 1, "c": 2, "e": 3}}
    norm = rrf([sparse, dense], [0.25, 0.75], K)["q"]
    raw = rrf([sparse, dense], [1.0, 3.0], K)["q"]
    assert list(raw) == list(norm)
    assert raw == pytest.approx(norm)
    assert norm["e"] == pytest.approx(0.75 / (K + 3))


@pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0, -0.5]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        rrf([{"q": {"a": 1}}, {"q": {"a": 1}}], weights, K)


def test_tie_order_is_stable(tmp_path):
    # With equal weights, a doc at rank r in one list only ties with a doc at rank r
    # in the other list only (the hybrid_best setting uses w_dense=0.5).
    sparse = {"q": {"a": 1, "m": 2}}
    dense = {"q": {"z": 1, "m": 3}}
    first = rrf([sparse, dense], [0.5, 0.5], K)
    assert first["q"]["a"] == first["q"]["z"]
    for _ in range(5):
        assert list(rrf([sparse, dense], [0.5, 0.5], K)["q"]) == list(first["q"])

    # A tie at the depth cutoff keeps the document from the first (sparse) list.
    assert set(rrf([sparse, dense], [0.5, 0.5], K, depth=2)["q"]) == {"m", "a"}

    # The written run lists tied documents in trec_eval order (docid descending).
    path = write_run(first, tmp_path / "fused.trec", tag="t")
    lines = [line.split() for line in path.read_text().splitlines()]
    assert [line[2] for line in lines] == ["m", "z", "a"]
    assert read_run(path)["q"].keys() == {"m", "z", "a"}

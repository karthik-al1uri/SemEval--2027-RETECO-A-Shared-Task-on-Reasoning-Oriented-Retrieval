import pytest

from reteco.runs import read_run, write_run


def test_round_trip(tmp_path):
    run = {"q1": {"a": 3.0, "b": 1.5, "c": 2.25}, "q2": {"x": 0.1}}
    p = write_run(run, tmp_path / "sub" / "r.trec", tag="test")
    assert read_run(p) == run
    lines = p.read_text().splitlines()
    assert lines[0] == "q1 Q0 a 1 3.000000 test"
    assert [l.split()[2] for l in lines[:3]] == ["a", "c", "b"]
    assert [l.split()[3] for l in lines[:3]] == ["1", "2", "3"]


def test_depth_cut(tmp_path):
    run = {"q": {f"d{i}": float(i) for i in range(250)}}
    p = write_run(run, tmp_path / "r.trec", tag="t", depth=100)
    got = read_run(p)["q"]
    assert len(got) == 100
    assert min(got.values()) == 150.0


def test_ties_follow_trec_eval_order(tmp_path):
    # trec_eval breaks score ties by docid descending; rank column must agree.
    run = {"q": {"a": 1.0, "c": 1.0, "b": 1.0000000001, "z": 2.0}}
    p = write_run(run, tmp_path / "r.trec", tag="t")
    assert [l.split()[2] for l in p.read_text().splitlines()] == ["z", "c", "b", "a"]


def test_bad_tag_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_run({"q": {"a": 1.0}}, tmp_path / "r.trec", tag="has space")


def test_malformed_and_duplicate_lines_rejected(tmp_path):
    p = tmp_path / "bad.trec"
    p.write_text("q Q0 a 1 1.0\n")
    with pytest.raises(ValueError):
        read_run(p)
    p.write_text("q Q0 a 1 1.0 t\nq Q0 a 2 0.5 t\n")
    with pytest.raises(ValueError):
        read_run(p)

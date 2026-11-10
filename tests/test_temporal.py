import pytest

from reteco.evaluate import evaluate_run
from reteco.temporal import doctimes
from reteco.temporal.extract import MIN_YEAR, PRESENT_YEAR, extract
from reteco.temporal.fit import FitParams, temporal_fit
from reteco.temporal.rescore import FastNDCG, build_table, combine, normalize, pair_features, to_run


def spans(text, **kw):
    return [(t.start, t.end, t.kind) for t in extract(text, **kw)]


@pytest.mark.parametrize("text,expected", [
    ("In 1944, were civilians able to buy jerrycans?", [(1944, 1944, "year")]),
    ("between 2000 and 2010", [(2000, 2010, "range")]),
    ("from 1914–18", [(1914, 1918, "range")]),
    ("the late 1950s", [(1957, 1959, "decade")]),
    ("in the 1800s", [(1800, 1899, "century")]),
    ("the 19th century", [(1800, 1899, "century")]),
    ("early nineteenth-century Britain", [(1800, 1832, "century")]),
    ("the 3rd century BC", [(-300, -201, "century")]),
    ("the 2nd millennium BC", [(-2000, -1001, "millennium")]),
    ("44 BC", [(-44, -44, "year")]),
    ("AD 79", [(79, 79, "year")]),
    ("476 AD", [(476, 476, "year")]),
    ("221 B.C.", [(-221, -221, "year")]),
    ("221–206 BC", [(-221, -206, "range")]),
    ("during WWII", [(1939, 1945, "era")]),
    ("in the Victorian era", [(1837, 1901, "era")]),
    ("sources prior to 1600", [(MIN_YEAR, 1600, "relative")]),
    ("pre-1600 sources", [(MIN_YEAR, 1600, "relative")]),
    ("since 1945", [(1945, PRESENT_YEAR, "relative")]),
    ("after WWII", [(1945, PRESENT_YEAR, "relative")]),
    ("On 5 March 1943", [(1943, 1943, "date")]),
    ("the '60s", [(1960, 1969, "decade")]),
])
def test_extract(text, expected):
    assert spans(text) == expected


@pytest.mark.parametrize("text", [
    "I sent 1500 BTC", "at block 2016", "it cost $1999", "version 1.2017", "1,500 people",
    "2000 dollars", "port 1080", "BIP 1234", "a 2 ADA deposit", "minimum of 25 ADA", "curves AD1 and AD2",
    "hash 9a d3 8c e1", "0x25BC",
])
def test_quantities_are_not_years(text):
    assert spans(text) == []


def test_present_only_on_request():
    assert spans("is it still used nowadays?") == []
    assert spans("nowadays", present=True) == [(PRESENT_YEAR - 4, PRESENT_YEAR, "present")]


def test_html_is_stripped():
    assert spans("<p>In <strong>1944</strong>&nbsp;they</p>") == [(1944, 1944, "year")]


def test_fit_neutral_overlap_and_distance():
    p = FitParams(tau=10, w_doc=1, w_q=1, w_best=0, center=0.25)
    q = [[1940, 1945, "era"]]
    assert temporal_fit(q, [], p) == 0.0                      # no doc time: neutral
    assert temporal_fit([], [[1900, 1900, "year"]], p) == 0.0  # no query time: neutral
    match = temporal_fit(q, [[1944, 1944, "year"]], p)
    near = temporal_fit(q, [[1950, 1950, "year"]], p)
    far = temporal_fit(q, [[1650, 1650, "year"]], p)
    assert match == pytest.approx(0.75)
    assert match > near > 0 > far
    assert far == pytest.approx(-0.25, abs=1e-6)
    boost = FitParams(tau=10, center=0.0)
    assert temporal_fit(q, [[1650, 1650, "year"]], boost) == pytest.approx(0.0, abs=1e-6)


def test_rescore_moves_time_matching_doc_up(tiny_data, monkeypatch, tmp_path):
    monkeypatch.setattr(doctimes, "DOC_CACHE", tmp_path / "cache")
    # q1 "When did the Chrysalis migration happen?" -> gazetteer era 2021; gold d1 (2021) ranked 2nd.
    # q4 "When was Black-Scholes published?" has no time: its order must not change.
    run = {"q1": {"iota/d2": 2.0, "iota/d1": 1.9, "iota/d3": 1.0},
           "q4": {"quant/d2": 2.0, "quant/d1": 1.0}}
    table = build_table(run, "train", root=tiny_data, workers=1)
    assert table.q_times[table.qids.index("q1")] == [[2021, 2021, "era"]]
    assert table.q_times[table.qids.index("q4")] == []
    p = FitParams(tau=5, center=0.25)
    temporal = pair_features(table, p).score(p)
    assert temporal.tolist() == pytest.approx([0.0, 0.75, 0.0, 0.0, 0.0])  # only q1/d1 has both
    scores = combine(table, normalize(table), temporal, 1.0, 0.5)
    new = to_run(table, scores)
    assert max(new["q1"], key=new["q1"].get) == "iota/d1"
    assert sorted(new["q4"], key=new["q4"].get, reverse=True) == ["quant/d2", "quant/d1"]
    # FastNDCG agrees with pytrec_eval
    ev = FastNDCG(table, "train", root=tiny_data)
    official = evaluate_run(new, "train", domains=["iota", "quant"], root=tiny_data)
    for d, r in official["domains"].items():
        assert ev.per_domain(scores)[d] == pytest.approx(r["ndcg@10"])

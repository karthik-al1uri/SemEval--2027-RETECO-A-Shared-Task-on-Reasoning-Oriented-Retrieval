import json

import pytest


@pytest.fixture
def tiny_data(tmp_path):
    """A two-domain Track 1 layout with a handful of documents and queries."""
    domains = {
        "iota": {
            "docs": [("iota/d1", "IOTA Chrysalis migration in 2021."),
                     ("iota/d2", "The legacy network was shut down."),
                     ("iota/d3", "Unrelated text about tangles.")],
            "train": [("q1", "When did the Chrysalis migration happen?", ["iota/d1"]),
                      ("q2", "What happened to the legacy network?", ["iota/d2", "iota/d1"])],
            "dev": [("q3", "Tell me about tangles.", ["iota/d3"])],
        },
        "quant": {
            "docs": [("quant/d1", "Black-Scholes was published in 1973."),
                     ("quant/d2", "Volatility smiles appeared after 1987.")],
            "train": [("q4", "When was Black-Scholes published?", ["quant/d1"])],
            "dev": [("q5", "When did volatility smiles appear?", ["quant/d2"])],
        },
    }
    for dom, spec in domains.items():
        d = tmp_path / dom
        d.mkdir()
        with open(d / "documents.jsonl", "w") as f:
            for i, c in spec["docs"]:
                f.write(json.dumps({"id": i, "content": c}) + "\n")
            f.write("\n")  # blank lines are skipped
        for split in ("train", "dev"):
            with open(d / f"examples_{split}.jsonl", "w") as ex, open(d / f"qrels_{split}.txt", "w") as qr:
                for qid, q, gold in spec[split]:
                    ex.write(json.dumps({"id": qid, "query": q, "gold_ids": gold,
                                         "gold_answers": ["secret"]}) + "\n")
                    for g in gold:
                        qr.write(f"{qid}\t0\t{g}\t1\n")
    return tmp_path

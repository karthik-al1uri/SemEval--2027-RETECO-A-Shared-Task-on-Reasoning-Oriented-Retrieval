"""Per-domain data statistics for the paper's data table.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_data_stats.py

Counts documents, train/dev queries, judged documents per query (qrels, used
here for description only, as in docs/DATA.md), query length after HTML
stripping, and the share of queries with an explicit time anchor (temporal
extractor, from results/analysis_query_types.csv).
Writes results/data_stats.csv.
"""
from __future__ import annotations

import csv
from statistics import mean, median

from reteco.data import DOMAINS, REPO_ROOT, data_dir, load_qrels, load_queries
from reteco.temporal.extract import clean_text

OUT = REPO_ROOT / "results" / "data_stats.csv"


def count_lines(path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n


def main() -> None:
    with open(REPO_ROOT / "results" / "analysis_query_types.csv", newline="") as f:
        anchored = {r["qid"]: int(r["anchored"]) for r in csv.DictReader(f)}
    rows = []
    for d in DOMAINS:
        qs = {s: load_queries(d, s) for s in ("train", "dev")}
        qrels = {**load_qrels(d, "train"), **load_qrels(d, "dev")}
        words = [len(clean_text(q.query).split()) for s in qs for q in qs[s]]
        allq = [q.id for s in qs for q in qs[s]]
        rows.append({
            "domain": d,
            "documents": count_lines(data_dir() / d / "documents.jsonl"),
            "train_queries": len(qs["train"]), "dev_queries": len(qs["dev"]),
            "gold_per_query": f"{mean(len(v) for v in qrels.values()):.2f}",
            "query_words_mean": f"{mean(words):.0f}", "query_words_median": f"{median(words):.0f}",
            "anchored_share": f"{mean(anchored[q] for q in allq):.3f}",
        })
        print(rows[-1])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

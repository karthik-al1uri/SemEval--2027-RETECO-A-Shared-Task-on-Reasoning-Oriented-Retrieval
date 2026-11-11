"""Shared helpers for the playground loop.

The internal split of TRAIN (configs/playground_split.json) has two parts:
  tune     ~80% of each domain's train queries: every parameter is tuned here
  holdout  ~20%: experiments are judged here only

Scores are macro nDCG@10 over the 13 domains restricted to one part, using
qrels_train (missing queries score 0, as in reteco.evaluate).

    python -m reteco.playground.common --make-split     (once; the file is committed)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data import DOMAINS, REPO_ROOT, load_qrels, load_queries
from ..evaluate import score_domain

SPLIT_PATH = REPO_ROOT / "configs" / "playground_split.json"
PG_CSV = REPO_ROOT / "results" / "playground.csv"
PG_FIELDS = ["id", "date", "experiment", "hypothesis", "change", "tune_ndcg@10", "holdout_ndcg@10",
             "delta_vs_champion", "holdout_no_iota", "latency_ms_per_query", "champion_latency_ms",
             "time_spent_min", "decision", "reason", "config"]


def make_split(seed: int = 13, frac: float = 0.2, min_holdout: int = 2) -> dict:
    """Stratified per domain: n_holdout = max(min_holdout, round(frac * n)), sampled with `seed`."""
    rng = random.Random(seed)
    out = {"seed": seed, "holdout_fraction": frac, "min_holdout_per_domain": min_holdout,
           "domains": {}}
    for d in DOMAINS:
        qids = sorted(q.id for q in load_queries(d, "train"))
        n_h = max(min_holdout, int(round(frac * len(qids))))
        hold = sorted(rng.sample(qids, n_h))
        out["domains"][d] = {"tune": [q for q in qids if q not in set(hold)], "holdout": hold}
    return out


@lru_cache(maxsize=1)
def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def part_qids(part: str, domains=None) -> set[str]:
    sp = load_split()["domains"]
    return {q for d in (domains or DOMAINS) for q in sp[d][part]}


def part_of(qid: str) -> str:
    for d, v in load_split()["domains"].items():
        if qid in v["tune"]:
            return "tune"
        if qid in v["holdout"]:
            return "holdout"
    raise KeyError(qid)


@lru_cache(maxsize=None)
def _qrels(domain: str) -> dict:
    return load_qrels(domain, "train")


def score_part(run: dict, part: str, domains=None) -> dict:
    """Macro nDCG@10 over `domains` restricted to the queries of one split part."""
    domains = list(domains or DOMAINS)
    keep = part_qids(part, domains)
    per = {}
    for d in domains:
        qrels = {q: r for q, r in _qrels(d).items() if q in keep}
        per[d] = score_domain(run, qrels)
    macro = float(np.mean([r["ndcg@10"] for r in per.values()]))
    no_iota = float(np.mean([r["ndcg@10"] for d, r in per.items() if d != "iota"]))
    return {"part": part, "macro": macro, "no_iota": no_iota,
            "recall@100": float(np.mean([r["recall@100"] for r in per.values()])),
            "domains": {d: r["ndcg@10"] for d, r in per.items()}}


def per_query_ndcg(run: dict, part: str, domains=None) -> dict[str, float]:
    import pytrec_eval
    out = {}
    keep = part_qids(part, domains)
    for d in (domains or DOMAINS):
        qrels = {q: r for q, r in _qrels(d).items() if q in keep}
        sub = {q: run[q] for q in qrels if q in run}
        ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(sub) if sub else {}
        for q in qrels:
            out[q] = ev.get(q, {}).get("ndcg_cut_10", 0.0)
    return out


def paired_bootstrap(a: dict, b: dict, n: int = 2000, seed: int = 13) -> tuple[float, float]:
    """95% CI of macro(b) - macro(a) over holdout queries, resampling within each domain."""
    rng = np.random.default_rng(seed)
    sp = load_split()["domains"]
    diffs = {d: np.array([b[q] - a[q] for q in sp[d]["holdout"]]) for d in DOMAINS}
    stats = []
    for _ in range(n):
        stats.append(np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in diffs.values()]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def log_row(row: dict) -> None:
    new = not PG_CSV.exists()
    row = {"date": dt.date.today().isoformat(), **row}
    with open(PG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PG_FIELDS})


def fmt(res: dict) -> str:
    dom = " ".join(f"{d[:4]}={v:.3f}" for d, v in res["domains"].items())
    return f"{res['part']:<8} macro={res['macro']:.4f} no_iota={res['no_iota']:.4f}  {dom}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.common")
    ap.add_argument("--make-split", action="store_true")
    args = ap.parse_args(argv)
    if args.make_split:
        if SPLIT_PATH.exists():
            raise SystemExit(f"{SPLIT_PATH} exists; the split is fixed")
        sp = make_split()
        SPLIT_PATH.write_text(json.dumps(sp, indent=1) + "\n")
        for d, v in sp["domains"].items():
            print(f"{d:<10} tune {len(v['tune']):>4}  holdout {len(v['holdout']):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

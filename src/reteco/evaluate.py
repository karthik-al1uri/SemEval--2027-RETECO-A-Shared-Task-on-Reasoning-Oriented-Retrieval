"""Score a TREC run with pytrec_eval: per-domain and macro nDCG@10 (+ Recall@100).

    python -m reteco.evaluate runs/dev/bm25.trec --split dev

Every query in a domain's qrels counts; a query missing from the run scores 0
(pytrec_eval alone would silently skip it). The macro average is the mean of
the per-domain means, as in the official RETECO baseline.

The scored result is appended to results/experiments.csv. The description and
config path come from --description/--config, or else from a sidecar file
<run>.meta.json written by the system that produced the run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import pytrec_eval

from .data import DOMAINS, REPO_ROOT, SPLITS, load_qrels
from .runs import read_run

METRICS = {"ndcg_cut.10", "recall.100"}
RESULTS_CSV = REPO_ROOT / "results" / "experiments.csv"
CSV_FIELDS = (["date", "system_name", "split", "n_domains", "macro_ndcg@10", "macro_recall@100"]
              + DOMAINS + ["description", "config", "latency_ms_per_query"])


def score_domain(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]) -> dict:
    """Mean nDCG@10 / Recall@100 over all queries in `qrels` (missing queries score 0)."""
    sub = {q: run[q] for q in qrels if q in run}
    per_query = pytrec_eval.RelevanceEvaluator(qrels, METRICS).evaluate(sub) if sub else {}
    n = len(qrels)
    ndcg = sum(per_query.get(q, {}).get("ndcg_cut_10", 0.0) for q in qrels) / n
    recall = sum(per_query.get(q, {}).get("recall_100", 0.0) for q in qrels) / n
    return {"ndcg@10": ndcg, "recall@100": recall, "num_queries": n,
            "missing_queries": n - len(sub)}


def evaluate_run(run: dict[str, dict[str, float]], split: str, domains=None, root=None) -> dict:
    """Score a run on the given domains (default: every domain the run covers)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    candidates = list(domains) if domains else DOMAINS
    all_qrels = {d: load_qrels(d, split, root) for d in candidates}
    if domains is None:
        domains = [d for d in candidates if any(q in run for q in all_qrels[d])]
        if not domains:
            raise ValueError(f"run has no queries from the {split} split")
    known = set().union(*all_qrels.values())
    unknown = [q for q in run if q not in known]
    per = {d: score_domain(run, all_qrels[d]) for d in domains}
    return {
        "split": split,
        "domains": per,
        "macro_ndcg@10": sum(r["ndcg@10"] for r in per.values()) / len(per),
        "macro_recall@100": sum(r["recall@100"] for r in per.values()) / len(per),
        "unknown_queries": len(unknown),
    }


def append_result(result: dict, system_name: str, description: str, config: str,
                  path: Path | None = None, latency_ms: float | None = None) -> Path:
    """Append one row to the results CSV. `latency_ms` is the measured per-query
    latency of the system's online stages (blank when not measured)."""
    path = Path(path) if path is not None else RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = {
        "date": dt.date.today().isoformat(),
        "system_name": system_name,
        "split": result["split"],
        "n_domains": len(result["domains"]),
        "macro_ndcg@10": f"{result['macro_ndcg@10']:.4f}",
        "macro_recall@100": f"{result['macro_recall@100']:.4f}",
        "description": description,
        "config": config,
        "latency_ms_per_query": "" if latency_ms is None else f"{latency_ms:.1f}",
    }
    for d in DOMAINS:
        r = result["domains"].get(d)
        row[d] = f"{r['ndcg@10']:.4f}" if r else ""
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return path


def format_table(result: dict) -> str:
    lines = [f"{'domain':<12}{'queries':>8}{'nDCG@10':>10}{'R@100':>9}"]
    for d, r in result["domains"].items():
        flag = f"  ({r['missing_queries']} missing)" if r["missing_queries"] else ""
        lines.append(f"{d:<12}{r['num_queries']:>8}{r['ndcg@10']:>10.4f}{r['recall@100']:>9.4f}{flag}")
    n = len(result["domains"])
    lines.append(f"{'MACRO':<12}{n:>7}d{result['macro_ndcg@10']:>10.4f}{result['macro_recall@100']:>9.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", choices=DOMAINS,
                    help="score only these domains (default: all domains in the run)")
    ap.add_argument("--name", help="system name (default: run file stem)")
    ap.add_argument("--description")
    ap.add_argument("--config")
    ap.add_argument("--no-log", action="store_true", help="do not append to results/experiments.csv")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    meta_path = args.run.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    result = evaluate_run(read_run(args.run), args.split, args.domains)

    print(json.dumps(result, indent=2) if args.json else format_table(result))
    if result["unknown_queries"]:
        print(f"warning: {result['unknown_queries']} run queries are not in the {args.split} qrels",
              file=sys.stderr)
    if not args.no_log:
        path = append_result(result, args.name or args.run.stem,
                             args.description or meta.get("description", ""),
                             args.config or meta.get("config", ""))
        print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

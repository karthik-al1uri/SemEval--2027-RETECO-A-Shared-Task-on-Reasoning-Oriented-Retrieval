"""Run a sparse system from a config, write TREC runs, score and log them.

    python -m reteco.sparse.run --config configs/sparse_bm25_global.yaml --splits train dev

Config keys:

    system_name   run file name and tag
    description   short text for results/experiments.csv
    corpus        raw | markup | clean               (default raw)
    query_clean   list of steps from reteco.sparse.clean.QUERY_STEPS (default none)
    k1, b         global BM25 parameters
    per_domain    optional {domain: {k1, b}} overriding the global ones
    rm3           optional {fb_docs, fb_terms, alpha, max_df}
    depth         run depth (100)

Per-domain results are cached under cache/sparse/runs/<config hash>/ so an
interrupted run resumes where it stopped. A (system, split, config) row is only
appended to results/experiments.csv once unless --relog is given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import yaml

from ..data import DOMAINS, REPO_ROOT, SPLITS, load_queries
from ..evaluate import RESULTS_CSV, append_result, evaluate_run, format_table
from ..runs import write_run
from .index import CACHE_DIR, TFIndex
from .retrieve import search, tokenize_queries


def domain_params(config: dict, domain: str) -> tuple[float, float]:
    p = (config.get("per_domain") or {}).get(domain, {})
    return float(p.get("k1", config["k1"])), float(p.get("b", config["b"]))


def config_key(config: dict, domain: str) -> str:
    k1, b = domain_params(config, domain)
    key = {"corpus": config.get("corpus", "raw"), "query_clean": list(config.get("query_clean") or []),
           "k1": k1, "b": b, "rm3": config.get("rm3"), "depth": config.get("depth", 100)}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def run_domain(config: dict, domain: str, splits, workers: int = 8) -> dict[str, dict]:
    """{split: {qid: {doc: score}}} for one domain, cached per (config, domain, split)."""
    cache = CACHE_DIR / "runs" / config_key(config, domain)
    out, todo = {}, []
    for split in splits:
        f = cache / f"{domain}_{split}.json"
        if f.exists():
            out[split] = json.loads(f.read_text())
        else:
            todo.append(split)
    if not todo:
        return out
    index = TFIndex.get(domain, config.get("corpus", "raw"), workers)
    k1, b = domain_params(config, domain)
    weights = index.bm25(k1, b)
    for split in todo:
        queries = load_queries(domain, split)
        toks = tokenize_queries(queries, config.get("query_clean"))
        ranked = search(index, toks, k1, b, config.get("depth", 100), config.get("rm3"), weights=weights)
        out[split] = {q.id: dict(r) for q, r in zip(queries, ranked)}
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{domain}_{split}.json").write_text(json.dumps(out[split]))
    return out


def already_logged(system: str, split: str, config_path: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return any(r["system_name"] == system and r["split"] == split and r["config"] == config_path
                   for r in csv.DictReader(f))


def run(config: dict, splits, domains, workers: int = 8, config_path: str = "",
        log: bool = True, relog: bool = False, out_dir: Path = REPO_ROOT / "runs") -> dict:
    runs = {s: {} for s in splits}
    for domain in domains:
        t0 = time.time()
        res = run_domain(config, domain, splits, workers)
        for s in splits:
            runs[s].update(res[s])
        print(f"  {domain:<10} {time.time() - t0:6.1f}s", flush=True)
    results = {}
    name = config["system_name"]
    for split in splits:
        path = write_run(runs[split], out_dir / split / f"{name}.trec", tag=name, depth=config.get("depth", 100))
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"description": config["description"], "config": config_path, "domains": list(domains)},
            indent=2) + "\n")
        result = evaluate_run(runs[split], split, list(domains))
        results[split] = result
        print(f"[{name} / {split}]")
        print(format_table(result))
        if log and list(domains) == DOMAINS and (relog or not already_logged(name, split, config_path)):
            append_result(result, name, config["description"], config_path)
            print("appended to results/experiments.csv")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.sparse.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--relog", action="store_true", help="log even if this system/split/config is logged")
    args = ap.parse_args(argv)
    config = yaml.safe_load(open(args.config))
    run(config, args.splits, args.domains, args.workers, args.config, not args.no_log, args.relog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-encoder reranking of a first-stage run, plus score interpolation.

    python -m reteco.rerank.run --config configs/rerank_bge_base.yaml
    python -m reteco.rerank.run --config configs/rerank_bge_base.yaml --domains small --depth 50

For each split (train first) it reranks the top `depth` candidates of
`first_stage` with the cross-encoder and writes two runs (top 100):

    runs/<split>/<name>.trec          reranker score alone
    runs/<split>/<name>_interp.trec   a * norm(first stage) + (1 - a) * norm(reranker),
                                      (a, norm) tuned on train (or fixed in the config)

<name> is system_name, plus _d<depth> when --depth overrides the config and
_small when only the 5 small domains are run. Both runs are scored and logged
to results/experiments.csv with the measured reranking latency per query
(cross-encoder wall time on this machine's MPS device; interpolation adds
well under a millisecond).
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import yaml

from ..data import DOMAINS, REPO_ROOT, SMALL_DOMAINS, SPLITS
from ..evaluate import append_result, evaluate_run, format_table
from ..runs import write_run
from ..sparse.run import already_logged
from . import CACHE_DIR
from . import candidates as C
from .cross_encoder import score_candidates
from .interpolate import A_GRID, NORMS, combine, tune


def resolve_domains(spec) -> list[str]:
    if spec in (None, "all"):
        return list(DOMAINS)
    if spec == "small" or spec == ["small"]:
        return list(SMALL_DOMAINS)
    return list(spec)


def log_run(run, split, name, desc, config_path, domains, latency_ms, extra, log, relog):
    path = write_run(run, REPO_ROOT / "runs" / split / f"{name}.trec", tag=name, depth=100)
    path.with_suffix(".meta.json").write_text(json.dumps(
        {"description": desc, "config": config_path, "domains": domains,
         "latency_ms_per_query": round(latency_ms, 1), **extra}, indent=2) + "\n")
    res = evaluate_run(run, split, domains)
    print(f"\n{name} [{split}]  latency {latency_ms:.0f} ms/query\n{format_table(res)}", flush=True)
    if log and (relog or not already_logged(name, split, config_path)):
        append_result(res, name, desc, config_path, latency_ms=latency_ms)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.rerank.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=None, help="all | small | domain names")
    ap.add_argument("--depth", type=int, help="override the config's reranking depth")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--relog", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    domains = resolve_domains(args.domains if args.domains else cfg.get("domains", "all"))
    depth = args.depth or cfg["depth"]
    name = cfg["system_name"]
    if args.depth:
        name += f"_d{depth}"
    if domains != list(DOMAINS):
        name += "_small" if domains == list(SMALL_DOMAINS) else "_" + "-".join(domains)
    splits = [s for s in SPLITS if s in args.splits]
    icfg = cfg.get("interpolate") or {}

    params = None if icfg.get("tune", True) else {"a": icfg["a"], "norm": icfg["norm"]}
    if params is None and "train" not in splits:
        raise SystemExit("interpolation is tuned on train: include the train split or fix (a, norm)")
    encoder = None
    for split in splits:
        print(f"== {name} [{split}] {cfg['model']} depth {depth} on {cfg['first_stage']} "
              f"({len(domains)} domains)", flush=True)
        t0 = time.time()
        cands = C.load(cfg["first_stage"], split, domains)
        scores, secs, encoder = score_candidates(cands, split, cfg, depth, encoder)
        latency = 1000 * float(np.mean(list(secs.values())))
        print(f"scored in {time.time() - t0:.0f}s; reranking latency {latency:.0f} ms/query", flush=True)
        base = (f"{cfg['description']}; top {depth} of {cfg['first_stage']}, "
                f"{cfg.get('dtype', 'float16')}, max_length {cfg.get('max_length', 512)}")
        extra = {"first_stage": cfg["first_stage"], "model": cfg["model"], "depth": depth}
        log_run(combine(cands, scores, depth, 0.0), split, name, f"{base}; reranker score only",
                args.config, domains, latency, extra, not args.no_log, args.relog)

        if split == "train" and params is None:
            t1 = time.time()
            best, rows = tune(cands, scores, depth, "train", domains,
                              icfg.get("a_grid", A_GRID), icfg.get("norms", NORMS))
            grid = CACHE_DIR / "interp" / f"{name}_grid_train.csv"
            grid.parent.mkdir(parents=True, exist_ok=True)
            with open(grid, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            params = {"a": best["a"], "norm": best["norm"]}
            print(f"interpolation tuned on train: a={best['a']} norm={best['norm']} "
                  f"nDCG@10={best['ndcg@10']:.4f} ({time.time() - t1:.0f}s; grid {grid})", flush=True)
        t1 = time.time()
        run = combine(cands, scores, depth, params["a"], params["norm"])
        lat_i = latency + 1000 * (time.time() - t1) / max(1, len(cands))
        log_run(run, split, f"{name}_interp",
                f"{base}; interpolated a={params['a']} norm={params['norm']} (a*first_stage + (1-a)*reranker"
                + (", tuned on train)" if icfg.get("tune", True) else ", fixed)"),
                args.config, domains, lat_i, {**extra, **params}, not args.no_log, args.relog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

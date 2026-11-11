"""Run a pipeline config on TRAIN and score it on the tune and holdout parts.

    python -m reteco.playground.harness --config configs/final_v1.yaml --name pg_champion

Writes runs/train/<name>.trec (+ .meta.json); does not touch results/experiments.csv
(playground rows go to results/playground.csv via reteco.playground.log). First-stage
outputs (sparse and dense depth-1000 lists and their timings) are cached under
cache/playground/first/ keyed by the sparse and dense configs, since every
experiment re-uses them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time

from ..data import DOMAINS, REPO_ROOT
from ..pipeline import Timer, latency, load_yaml, run_pipeline
from ..runs import write_run
from .common import fmt, part_qids, score_part

CACHE = REPO_ROOT / "cache" / "playground"


def _first_key(cfg: dict, stage: str) -> str:
    sub = load_yaml(cfg[stage]["config"])
    extra = cfg.get(stage + "_extra", {})
    spec = {"cfg": sub, "extra": extra, "depth": cfg["fusion"].get("depth", 1000)}
    if cfg.get("query_shorten"):
        spec["query_shorten"] = cfg["query_shorten"]
    spec = json.dumps(spec,
                      sort_keys=True, default=str)
    return f"{stage}_{hashlib.sha1(spec.encode()).hexdigest()[:10]}"


def train_candidates(cfg: dict, depth: int | None = None) -> dict:
    """Fusion candidate lists for every train query (first stage from the cache)."""
    from ..pipeline import dense_stage, fusion_stage, sparse_stage
    cfg.setdefault("components", {})
    shared = preload_shared(cfg)
    first = {}
    for stage, fn in (("sparse", sparse_stage), ("dense", dense_stage)):
        key = (stage, tuple(DOMAINS))
        if key not in shared:
            t = Timer()
            args = (cfg, DOMAINS, ["train"], cfg["fusion"].get("depth", 1000), t)
            shared[key] = (fn(*args), t)
        first[stage] = shared[key][0]["train"]
    save_shared(cfg, shared)
    cands = fusion_stage(cfg, first["sparse"], first["dense"], cfg["fusion"].get("depth", 1000))
    return {q: v[:depth] for q, v in cands.items()} if depth else cands


def preload_shared(cfg: dict, split: str = "train") -> dict:
    """shared dict for run_pipeline with cached first-stage outputs (if present)."""
    shared = {}
    for stage in ("sparse", "dense"):
        if not cfg["components"].get(stage, True):
            continue
        p = CACHE / "first" / f"{_first_key(cfg, stage)}_{split}.pkl"
        if p.exists():
            runs_, stages = pickle.load(open(p, "rb"))
            t = Timer()
            t.stages = stages
            shared[(stage, tuple(DOMAINS))] = (runs_, t)
    return shared


def save_shared(cfg: dict, shared: dict, split: str = "train") -> None:
    (CACHE / "first").mkdir(parents=True, exist_ok=True)
    for stage in ("sparse", "dense"):
        key = (stage, tuple(DOMAINS))
        if key in shared and cfg["components"].get(stage, True):
            p = CACHE / "first" / f"{_first_key(cfg, stage)}_{split}.pkl"
            if not p.exists():
                runs_, t = shared[key]
                pickle.dump(({split: runs_[split]}, t.stages), open(p, "wb"))


def run_config(cfg: dict, name: str, split: str = "train", write: bool = True, verbose: bool = True,
               parts=("tune", "holdout"), qids: set[str] | None = None):
    """Run `cfg` on `split`. On train, only the queries of `parts` are run and scored
    (or exactly `qids`, when given)."""
    cfg.setdefault("components", {})
    shared = preload_shared(cfg, split)
    t0 = time.time()
    if split == "train" and qids is None and set(parts) != {"tune", "holdout"}:
        qids = set().union(*(part_qids(p) for p in parts))
    runs, timer = run_pipeline(cfg, [split], DOMAINS, 8, shared, qids=qids)
    save_shared(cfg, shared, split)
    run = runs[split]
    lat = latency(timer, split, len(run))
    if qids is not None:
        # the first stage and fusion ran for every query of the split: rescale their time
        n_all = len(next(iter(v[0][split] for k, v in shared.items() if isinstance(k, tuple))))
        for st in ("sparse", "dense", "fusion"):
            if st in lat:
                lat[st] = round(lat[st] * len(run) / n_all, 1)
        lat["total"] = round(sum(v for k, v in lat.items() if k != "total"), 1)
    if write:
        path = write_run(run, REPO_ROOT / "runs" / split / f"{name}.trec", tag=name, depth=100)
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"config": cfg.get("_path", ""), "latency_ms_per_query": lat}, indent=2) + "\n")
    out = {"latency": lat, "seconds": time.time() - t0, "run": run}
    if split == "train":
        for p in parts:
            out[p] = score_part(run, p)
            if verbose:
                print(fmt(out[p]))
    if verbose:
        print(f"latency ms/query {lat}  ({out['seconds']:.0f}s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="+", default=["tune", "holdout"], choices=["tune", "holdout"])
    ap.add_argument("--qset", choices=["tune_train", "tune_val"], default=None,
                    help="run and score only this sub-part of tune instead of --parts")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    cfg["_path"] = args.config
    if args.qset:
        from .common import fmt, score_set, subsplit_tune
        qs = dict(zip(("tune_train", "tune_val"), subsplit_tune()))[args.qset]
        out = run_config(cfg, args.name, parts=(), qids=qs, write=False, verbose=False)
        print(fmt(score_set(out["run"], qs, args.qset)))
        return 0
    run_config(cfg, args.name, parts=args.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

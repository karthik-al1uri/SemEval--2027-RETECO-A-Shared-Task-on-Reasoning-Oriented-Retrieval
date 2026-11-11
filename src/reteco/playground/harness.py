"""Run a pipeline config on TRAIN and score it on the tune and holdout parts.

    python -m reteco.playground.harness --config configs/final.yaml --name pg_champion

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
from .common import fmt, score_part

CACHE = REPO_ROOT / "cache" / "playground"


def _first_key(cfg: dict, stage: str) -> str:
    sub = load_yaml(cfg[stage]["config"])
    extra = cfg.get(stage + "_extra", {})
    spec = json.dumps({"cfg": sub, "extra": extra, "depth": cfg["fusion"].get("depth", 1000)},
                      sort_keys=True, default=str)
    return f"{stage}_{hashlib.sha1(spec.encode()).hexdigest()[:10]}"


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


def run_config(cfg: dict, name: str, split: str = "train", write: bool = True, verbose: bool = True):
    cfg.setdefault("components", {})
    shared = preload_shared(cfg, split)
    t0 = time.time()
    runs, timer = run_pipeline(cfg, [split], DOMAINS, 8, shared)
    save_shared(cfg, shared, split)
    run = runs[split]
    lat = latency(timer, split, len(run))
    if write:
        path = write_run(run, REPO_ROOT / "runs" / split / f"{name}.trec", tag=name, depth=100)
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"config": cfg.get("_path", ""), "latency_ms_per_query": lat}, indent=2) + "\n")
    out = {"latency": lat, "seconds": time.time() - t0, "run": run}
    if split == "train":
        out["tune"] = score_part(run, "tune")
        out["holdout"] = score_part(run, "holdout")
        if verbose:
            print(fmt(out["tune"]))
            print(fmt(out["holdout"]))
            print(f"latency ms/query {lat}  ({out['seconds']:.0f}s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    cfg["_path"] = args.config
    run_config(cfg, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

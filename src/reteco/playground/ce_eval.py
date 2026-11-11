"""Score a cross-encoder variant inside the final pipeline and tune (a, norm) on a query set.

    python -m reteco.playground.ce_eval --model cache/playground/ce/minilm_ft/epoch1 --tune-on tune_val

`--tune-on` is a train part (tune) or tune_val (the 20% of tune that a model trained on
tune_train has not seen). The interpolation weight is chosen like configs/final.yaml's
(macro nDCG@10 of the interpolated run, grid a in {0, .05, ..., 1} x {minmax, zscore},
ties to the smaller a), but on that set only. The full chain (interp -> temporal ->
dedup) is then scored on the tuning set and, with --holdout, on holdout.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json

from ..pipeline import load_yaml
from ..rerank.cross_encoder import score_candidates
from ..rerank.interpolate import A_GRID, NORMS, combine
from .common import fmt, part_qids, score_set, subsplit_tune
from .harness import run_config, train_candidates


def qset(name: str) -> set[str]:
    if name in ("tune", "holdout"):
        return part_qids(name)
    tr, va = subsplit_tune()
    return {"tune_train": tr, "tune_val": va}[name]


def tune_interp(cfg: dict, qids: set[str], label: str, verbose: bool = True):
    rcfg = dict(cfg["rerank"], seed=13)
    depth = rcfg["depth"]
    cands = {q: v for q, v in train_candidates(cfg, 1000).items() if q in qids}
    scores, secs, _ = score_candidates(cands, "train", rcfg, depth)
    rows = []
    for method, a in itertools.product(NORMS, A_GRID):
        if a == 0 and method != NORMS[0]:
            continue
        r = score_set(combine(cands, scores, depth, a, method), qids, label)
        rows.append({"a": a, "norm": method, "ndcg": r["macro"]})
    best = max(rows, key=lambda r: (round(r["ndcg"], 6), -r["a"], r["norm"] == "minmax"))
    alone = next(r["ndcg"] for r in rows if r["a"] == 0)
    if verbose:
        print(f"[{label}] CE alone {alone:.4f}; best interp a={best['a']} {best['norm']} {best['ndcg']:.4f}")
    return best, rows, cands, scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.ce_eval")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--model", default=None, help="override rerank.model")
    ap.add_argument("--set", nargs="*", default=[], help="rerank overrides key=json_value")
    ap.add_argument("--tune-on", default="tune", choices=["tune", "tune_val", "tune_train"])
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--name", default="pg_ce_eval")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    cfg = copy.deepcopy(cfg)
    if args.model:
        cfg["rerank"]["model"] = args.model
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg["rerank"][k] = json.loads(v)
    qs = qset(args.tune_on)
    best, rows, _, _ = tune_interp(cfg, qs, args.tune_on)
    cfg["interpolate"] = {"a": best["a"], "norm": best["norm"]}
    full = run_config(cfg, args.name + "_" + args.tune_on, parts=(), qids=qs, write=False, verbose=False)
    print(fmt(score_set(full["run"], qs, args.tune_on + "_full")))
    if args.holdout:
        res = run_config(cfg, args.name, parts=("holdout",))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

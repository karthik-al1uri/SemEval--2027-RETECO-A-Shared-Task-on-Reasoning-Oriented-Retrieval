"""Grid over config values, full pipeline scored on one query set (a train part or tune_val).

    python -m reteco.playground.grid --config configs/pg_e1.yaml --qset tune_val \
        --param fusion.k=10,20,40 --param fusion.w_dense=0.4,0.5,0.6

Values are parsed as YAML. Prints one line per grid point and the best (ties: first in grid order).
"""
from __future__ import annotations

import argparse
import copy
import itertools

import yaml

from ..pipeline import load_yaml
from .ce_eval import qset
from .common import score_set
from .harness import run_config


def set_key(cfg: dict, dotted: str, value):
    *path, last = dotted.split(".")
    node = cfg
    for p in path:
        node = node.setdefault(p, {})
    node[last] = value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.grid")
    ap.add_argument("--config", required=True)
    ap.add_argument("--qset", default="tune_val", choices=["tune", "tune_val", "tune_train"])
    ap.add_argument("--param", action="append", required=True, help="dotted.key=v1,v2,...")
    args = ap.parse_args(argv)
    base = load_yaml(args.config)
    qs = qset(args.qset)
    keys, vals = [], []
    for p in args.param:
        k, v = p.split("=", 1)
        keys.append(k)
        vals.append([yaml.safe_load(x) for x in v.split(",")])
    rows = []
    for combo in itertools.product(*vals):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            set_key(cfg, k, v)
        out = run_config(cfg, "pg_grid", parts=(), qids=qs, write=False, verbose=False)
        r = score_set(out["run"], qs, args.qset)
        rows.append((combo, r["macro"], r["no_iota"], out["latency"]["total"]))
        print(f"GRID {dict(zip(keys, combo))}  {r['macro']:.4f}  no_iota {r['no_iota']:.4f}  "
              f"{out['latency']['total']:.0f} ms", flush=True)
    best = max(rows, key=lambda r: round(r[1], 6))
    print(f"BEST {dict(zip(keys, best[0]))}  {best[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Grid-search the temporal rescorer on TRAIN (macro nDCG@10) and write configs.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo \\
    python -m reteco.temporal.tune --run runs/train/bm25.trec

Each named variant has its own search space; the best setting on train is
written to configs/temporal_<variant>.yaml. Dev is never touched here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

from pathlib import Path

import numpy as np
import yaml

from ..data import REPO_ROOT
from ..runs import read_run
from .doctimes import CACHE_DIR
from .fit import FitParams
from .rescore import FastNDCG, build_table, combine, normalize, pair_features

B_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
TAUS = [3.0, 10.0, 25.0, 50.0, 100.0]
WEIGHTS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0), (1, 2, 0)]
CENTERS = [0.0, 0.25, 0.5, 0.75, 1.0]

VARIANTS = {
    # name: (description, overrides of the search space)
    "boost": ("temporal-fit boost only (center=0): time-matching docs move up, others unchanged",
              {"centers": [0.0]}),
    "fit": ("temporal-fit, center tuned (center=1 means penalty only: wrong-time docs drop, no-time docs neutral)",
            {}),
    "fit_years": ("ablation: temporal-fit using only explicit dates/years/ranges/decades/centuries (no eras, no relative)",
                  {"drop_kinds": ("era", "relative", "present")}),
    "fit_present": ("temporal-fit + query 'nowadays/currently/today' mapped to 2022-2026",
                    {"query_present": True}),
    "fit_gated": ("temporal-fit with b scaled by the guidance-trained calendar-anchor classifier probability",
                  {"gate": {"C": 3.0}}),
    "fit_gated_hard": ("temporal-fit applied only to queries the guidance-trained classifier labels calendar-anchored (p>=0.5)",
                       {"gate": {"C": 3.0, "threshold": 0.5}}),
}


class Tuner:
    def __init__(self, run_path: Path, split: str = "train", workers: int = 8):
        assert split == "train", "tune on train only"
        self.run = read_run(run_path)
        self.split = split
        self.workers = workers
        self._tables = {}
        self._feats = {}
        self._norm = {}
        self._eval = {}

    def table(self, present: bool):
        if present not in self._tables:
            t = build_table(self.run, self.split, 100, present, self.workers)
            self._tables[present] = t
            self._eval[present] = FastNDCG(t, self.split)
        return self._tables[present], self._eval[present]

    def feats(self, present: bool, drop_kinds: tuple, max_width):
        key = (present, drop_kinds, max_width)
        if key not in self._feats:
            t, _ = self.table(present)
            self._feats[key] = pair_features(t, FitParams(drop_kinds=drop_kinds, max_width=max_width))
        return self._feats[key]

    def norm(self, present: bool, method: str):
        key = (present, method)
        if key not in self._norm:
            self._norm[key] = normalize(self.table(present)[0], method)
        return self._norm[key]

    def search(self, space: dict) -> dict:
        present = space.get("query_present", False)
        drop = tuple(space.get("drop_kinds", ()))
        widths = space.get("max_widths", [None, 500])
        norms = space.get("norms", ["minmax", "zscore"])
        centers = space.get("centers", CENTERS)
        gate_cfg = space.get("gate")
        table, ev = self.table(present)
        gate = None
        if gate_cfg:
            from .classify import predict_gate

            gate = predict_gate(table, self.split, gate_cfg)
            if gate_cfg.get("threshold") is not None:
                gate = (gate >= gate_cfg["threshold"]).astype(float)
        base = ev.macro(table.retrieval)
        best = {"macro": base, "params": None}
        n = 0
        for width in widths:
            pf = self.feats(present, drop, width)
            for tau in TAUS:
                comps = pf.components(tau)
                for (wd, wq, wb), center in itertools.product(WEIGHTS, centers):
                    sim = (wd * comps[0] + wq * comps[1] + wb * comps[2]) / (wd + wq + wb)
                    temporal = np.where(pf.has, sim - center, 0.0)
                    for method in norms:
                        nm = self.norm(present, method)
                        for b in B_GRID[1:]:
                            m = ev.macro(combine(table, nm, temporal, 1.0, b, gate))
                            n += 1
                            if m > best["macro"] + 1e-9:
                                best = {"macro": m, "params": {
                                    "tau": tau, "w_doc": wd, "w_q": wq, "w_best": wb, "center": center,
                                    "max_width": width, "norm": method, "b": b}}
        best["baseline"] = base
        best["evaluated"] = n
        if best["params"]:
            p = best["params"]
            fp = FitParams(tau=p["tau"], w_doc=p["w_doc"], w_q=p["w_q"], w_best=p["w_best"],
                           center=p["center"], drop_kinds=drop, max_width=p["max_width"])
            temporal = pair_features(table, fp).score(fp)
            nm = self.norm(present, p["norm"])
            best["per_domain"] = ev.per_domain(combine(table, nm, temporal, 1.0, p["b"], gate))
            best["b_curve"] = {b: ev.macro(combine(table, nm, temporal, 1.0, b, gate)) for b in B_GRID}
        best["baseline_per_domain"] = ev.per_domain(table.retrieval)
        return best


def write_config(name: str, desc: str, space: dict, best: dict) -> Path:
    p = best["params"] or {"tau": 10.0, "w_doc": 1, "w_q": 1, "w_best": 0, "center": 0.0,
                           "max_width": None, "norm": "minmax", "b": 0.0}
    tuned = (f"tuned on train: tau={p['tau']:g} w=({p['w_doc']},{p['w_q']},{p['w_best']}) "
             f"center={p['center']:g} max_width={p['max_width']} norm={p['norm']} b={p['b']:g}")
    cfg = {
        "system_name": f"temporal_{name}",
        "description": f"{desc}; {tuned}",
        "depth": 100,
        "seed": 13,
        "norm": p["norm"],
        "a": 1.0,
        "b": p["b"],
        "query_present": bool(space.get("query_present", False)),
        "fit": {"tau": p["tau"], "w_doc": p["w_doc"], "w_q": p["w_q"], "w_best": p["w_best"],
                "center": p["center"], "drop_kinds": list(space.get("drop_kinds", ())),
                "max_width": p["max_width"]},
        "tuned_on": {"split": "train", "input_run": "runs/train/bm25.trec",
                     "train_macro_ndcg@10": round(best["macro"], 4),
                     "train_baseline": round(best["baseline"], 4)},
    }
    if space.get("gate"):
        cfg["gate"] = dict(space["gate"])
    path = REPO_ROOT / "configs" / f"temporal_{name}.yaml"
    path.write_text("# Tuned on train by `python -m reteco.temporal.tune`.\n"
                    + yaml.safe_dump(cfg, sort_keys=False))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.temporal.tune")
    ap.add_argument("--run", type=Path, default=Path("runs/train/bm25.trec"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="rewrite configs from cached search results")
    args = ap.parse_args(argv)
    tuner = None if args.reuse else Tuner(args.run, "train", args.workers)
    out_dir = CACHE_DIR / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        desc, space = VARIANTS[name]
        t0 = time.time()
        cached = out_dir / f"{args.run.stem}_{name}.json"
        best = json.loads(cached.read_text()) if args.reuse else tuner.search(space)
        path = write_config(name, desc, space, best)
        (out_dir / f"{args.run.stem}_{name}.json").write_text(json.dumps(best, indent=2, default=str))
        print(f"{name:<16} train macro {best['macro']:.4f} (baseline {best['baseline']:.4f}) "
              f"params {best['params']}  [{best['evaluated']} settings, {time.time() - t0:.0f}s] -> {path.name}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

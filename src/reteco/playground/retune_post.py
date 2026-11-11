"""E9: re-tune the temporal rescorer and near-duplicate demotion on top of a new champion.

    python -m reteco.playground.retune_post --config configs/pg_e1.yaml --tune-on tune_val --out configs/pg_e9.yaml

1. Run the champion up to interpolation (temporal and dedup off) on the tuning queries.
2. Temporal: the full grid of reteco.temporal.tune for the variants boost, fit, fit_years and
   fit_present on that run; the variant with the best macro nDCG@10 wins (the run without
   temporal, b = 0, is the baseline every variant must beat).
3. Dedup: thresholds of reteco.analysis_tools.dedup x skip {0, 1} (plus "off") on the output of 2.

Macro scores are over the tuning queries only (the temporal tuner's per-domain divisor is
patched to the number of tuning queries per domain).
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter

import yaml

from ..analysis_tools.dedup import THRESHOLDS, apply, profile
from ..analysis_tools.errors import fetch_docs
from ..data import REPO_ROOT
from ..pipeline import load_yaml, temporal_stage, Timer
from ..runs import _ordered
from ..temporal.tune import VARIANTS, Tuner, write_config
from .ce_eval import qset
from .common import score_set
from .harness import run_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.retune_post")
    ap.add_argument("--config", default="configs/pg_e1.yaml")
    ap.add_argument("--tune-on", default="tune_val", choices=["tune", "tune_val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["boost", "fit", "fit_years", "fit_present"])
    args = ap.parse_args(argv)
    base = load_yaml(args.config)
    qs = qset(args.tune_on)
    stem = args.out.rsplit("/", 1)[-1].removesuffix(".yaml")

    # 1. interpolated run on the tuning queries
    c = copy.deepcopy(base)
    c["components"].update(temporal=False, dedup=False)
    name = f"{stem}_interp_{args.tune_on}"
    out = run_config(c, name, parts=(), qids=qs, write=True, verbose=False)
    interp = out["run"]
    print(f"interp only: {score_set(interp, qs, 'x')['macro']:.4f}")

    # 2. temporal
    tuner = Tuner(REPO_ROOT / "runs" / "train" / f"{name}.trec")
    n = Counter()
    for q in qs:
        n[next(iter(interp[q])).split("/", 1)[0]] += 1
    for present in (False, True):
        _, ev = tuner.table(present)
        ev.domain_n = {d: n[d] for d in ev.domain_n}
    results = {}
    for v in args.variants:
        desc, space = VARIANTS[v]
        best = tuner.search(space)
        results[v] = (desc, space, best)
        print(f"temporal {v}: {best['macro']:.4f} (baseline {best['baseline']:.4f}) {best['params']}")
    v = max(results, key=lambda k: results[k][2]["macro"])
    desc, space, best = results[v]
    if best["params"] is None:
        temporal_cfg = None
        t_run = interp
        print("temporal: no variant beats the run without temporal -> temporal off")
    else:
        path = write_config(v, desc, space, best, prefix=f"{stem}_temporal_",
                            input_run=f"runs/train/{name}.trec")
        temporal_cfg = str(path.relative_to(REPO_ROOT))
        c2 = dict(base, temporal={"config": temporal_cfg})
        t_run = temporal_stage(c2, interp, "train", Timer())
        print(f"temporal {v} -> {score_set(t_run, qs, 'x')['macro']:.4f}  ({temporal_cfg})")

    # 3. dedup
    run100 = {q: dict(_ordered(docs, 100)) for q, docs in t_run.items()}
    prof = profile(run100, texts=fetch_docs({d for docs in run100.values() for d in docs}))
    rows = [("off", None, None, score_set(run100, qs, "x")["macro"])]
    for skip in (0, 1):
        for t in THRESHOLDS:
            r, _ = apply(prof, t, skip)
            rows.append((f"t={t} skip={skip}", t, skip, score_set(r, qs, "x")["macro"]))
    for r in rows:
        print(f"  dedup {r[0]:<18} {r[3]:.4f}")
    bestd = max(rows, key=lambda r: round(r[3], 6))
    print(f"dedup best: {bestd[0]} {bestd[3]:.4f}")

    cfg = copy.deepcopy(base)
    cfg["system_name"] = stem
    cfg["components"]["temporal"] = temporal_cfg is not None
    if temporal_cfg:
        cfg["temporal"] = {"config": temporal_cfg}
    if bestd[1] is None and bestd[0] == "off":
        cfg["components"]["dedup"] = False
    else:
        dpath = REPO_ROOT / "configs" / f"{stem}_dedup.yaml"
        dpath.write_text("# Tuned by reteco.playground.retune_post on the " + args.tune_on + " part of train.\n"
                         + yaml.safe_dump({"system_name": f"{stem}_dedup", "threshold": bestd[1],
                                           "skip": bestd[2], "depth": 100, "seed": 13,
                                           "tuned_on": {"split": "train", "part": args.tune_on,
                                                        "macro_ndcg@10": round(bestd[3], 4)}},
                                          sort_keys=False))
        cfg["dedup"] = {"config": str(dpath.relative_to(REPO_ROOT))}
    with open(REPO_ROOT / args.out, "w") as f:
        f.write(f"# Playground E9: temporal and dedup re-tuned on {args.tune_on} on top of {args.config}.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

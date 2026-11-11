"""E14: ensemble of the fine-tuned and the original MiniLM cross-encoders.

    python -m reteco.playground.e14_ensemble --config configs/pg_e8.yaml

final = w_f * z(fusion) + w_ft * z(CE_finetuned) + w_orig * z(CE_original) over the reranked
top `depth`, z = per-query z-score (the champion's norm). (w_f, w_ft, w_orig) on a 0.05 grid
summing to 1 is tuned on tune_val (interpolated run, like the champion's a), then the full
chain (temporal, dedup) is scored on tune_val and holdout. Latency = the champion's pipeline
plus the original cross-encoder's measured time for the same pairs.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np

from ..pipeline import Timer, dedup_stage, load_yaml, temporal_stage
from ..rerank.cross_encoder import score_candidates
from ..rerank.interpolate import norm
from .ce_eval import qset
from .common import fmt, score_set
from .harness import train_candidates

ORIG = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def combine3(cands, s_ft, s_or, depth, w, out_depth=100):
    out = {}
    for q, docs in cands.items():
        top = docs[:depth]
        f = np.array([s for _, s in top])
        a = np.array([s_ft[q][d] for d, _ in top])
        b = np.array([s_or[q][d] for d, _ in top])
        sc = w[0] * norm(f, "zscore") + w[1] * norm(a, "zscore") + w[2] * norm(b, "zscore")
        run = {d: float(s) for (d, _), s in zip(top, sc)}
        low = float(sc.min()) - 1.0
        for i, (d, _) in enumerate(docs[depth:out_depth]):
            run[d] = low - 1e-3 * (i + 1)
        out[q] = dict(sorted(run.items(), key=lambda x: -x[1])[:out_depth])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.e14_ensemble")
    ap.add_argument("--config", default="configs/pg_e8.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    depth = cfg["rerank"]["depth"]
    r_ft = dict(cfg["rerank"], seed=13)
    r_or = dict(cfg["rerank"], model=ORIG, seed=13)
    cands_all = train_candidates(cfg, 1000)

    def scores(qs):
        c = {q: v for q, v in cands_all.items() if q in qs}
        a, sa, _ = score_candidates(c, "train", r_ft, depth)
        b, sb, _ = score_candidates(c, "train", r_or, depth)
        return c, a, b, sa, sb

    val = qset("tune_val")
    c, a, b, _, _ = scores(val)
    grid = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
    rows = []
    for wf in grid:
        for wft in grid:
            wo = round(1 - wf - wft, 2)
            if wo < -1e-9:
                continue
            rows.append(((wf, wft, wo), score_set(combine3(c, a, b, depth, (wf, wft, wo)), val, "v")["macro"]))
    best = max(rows, key=lambda r: (round(r[1], 6), -r[0][2]))  # ties: less weight on the extra model
    champ = next(s for w, s in rows if w == (cfg["interpolate"]["a"], round(1 - cfg["interpolate"]["a"], 2), 0.0))
    print(f"tune_val interp: champion (w_orig=0) {champ:.4f}; best {best[0]} {best[1]:.4f}")
    w = best[0]

    def chain(c, a, b, label, qs):
        run = combine3(c, a, b, depth, w)
        t = Timer()
        run = temporal_stage(cfg, run, "train", t)
        run = dedup_stage(cfg, run, "train", t)
        r = score_set(run, qs, label)
        print(fmt(r))
        return run

    chain(c, a, b, "tune_val", val)
    hold = qset("holdout")
    c, a, b, sa, sb = scores(hold)
    from ..runs import write_run
    from ..data import REPO_ROOT
    run = chain(c, a, b, "holdout", hold)
    write_run(run, REPO_ROOT / "runs" / "train" / "pg_e14.trec", tag="pg_e14", depth=100)
    print(f"CE ms/query: fine-tuned {1000 * np.mean(list(sa.values())):.1f}, "
          f"original {1000 * np.mean(list(sb.values())):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

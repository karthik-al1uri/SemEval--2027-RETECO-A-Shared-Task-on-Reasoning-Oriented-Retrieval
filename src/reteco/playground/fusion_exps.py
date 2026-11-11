"""E2 (per-domain interpolation weight) and E4 (three-way RRF) on cached cross-encoder scores.

    python -m reteco.playground.fusion_exps per_domain --tune-on tune --out configs/pg_e2.yaml
    python -m reteco.playground.fusion_exps rrf3 --tune-on tune --out configs/pg_e4.yaml

Everything is tuned on the given query set (a train part); holdout is only scored by
the harness afterwards.

per_domain: the global (a, norm) is re-tuned on the set; then each domain gets its own a
(norm fixed to the global one), shrunk toward the global a: a_d = (n_d a*_d + m a) / (n_d + m),
where n_d is the domain's number of tuning queries. The shrinkage m in {0, 5, 10, 20, 50, inf}
is chosen by 5-fold cross-validation inside the tuning set (inf = global a only).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import random

import numpy as np
import yaml

from ..data import DOMAINS, REPO_ROOT
from ..pipeline import load_yaml
from ..rerank.cross_encoder import score_candidates
from ..rerank.interpolate import A_GRID, NORMS, combine
from .ce_eval import qset
from .common import score_set
from .harness import preload_shared, train_candidates

M_GRID = [0, 5, 10, 20, 50, float("inf")]


def rrf3(cands, rerank, depth: int, sparse_run, dense_run, k: float, weights,
         out_depth: int = 100):
    """Weighted RRF of sparse, dense and cross-encoder ranks over the reranked top `depth`
    of the fusion list: sum_i w_i / (k + rank_i(d)); a list that misses d adds 0.
    Documents below `depth` keep their fusion order underneath (like `combine`)."""
    ws, wd, wc = weights
    out = {}
    for q, docs in cands.items():
        top = [d for d, _ in docs[:depth]]
        if not top:
            out[q] = {}
            continue
        sr = {d: r for r, d in enumerate(_rank(sparse_run.get(q, {})), 1)} if sparse_run else {}
        dr = {d: r for r, d in enumerate(_rank(dense_run.get(q, {})), 1)} if dense_run else {}
        cr = {d: r for r, d in enumerate(sorted(top, key=lambda d: -rerank[q][d]), 1)}
        run = {}
        for d in top:
            s = wc / (k + cr[d])
            if d in sr:
                s += ws / (k + sr[d])
            if d in dr:
                s += wd / (k + dr[d])
            run[d] = s
        low = min(run.values())
        for i, (d, _) in enumerate(docs[depth:out_depth]):
            run[d] = low - 1e-6 * (i + 1)
        out[q] = dict(sorted(run.items(), key=lambda x: -x[1])[:out_depth])
    return out


def _rank(docs: dict[str, float]) -> list[str]:
    """trec_eval order: score descending, ties by docid descending."""
    order = sorted(docs.items(), key=lambda x: x[0], reverse=True)
    order.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in order]



def dom(q_docs) -> str:
    return q_docs[0][0].split("/", 1)[0]


def perq(run, qids):
    """Per-query nDCG@10 for the given train qids."""
    from .common import _qrels
    import pytrec_eval
    out = {}
    for d in DOMAINS:
        qrels = {q: r for q, r in _qrels(d).items() if q in qids}
        if not qrels:
            continue
        sub = {q: run[q] for q in qrels if q in run}
        ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(sub)
        out.update({q: ev.get(q, {}).get("ndcg_cut_10", 0.0) for q in qrels})
    return out


def per_domain(cfg, qids, depth):
    cands = {q: v for q, v in train_candidates(cfg, 1000).items() if q in qids}
    scores, _, _ = score_candidates(cands, "train", dict(cfg["rerank"], seed=13), depth)
    qdom = {q: dom(v) for q, v in cands.items()}
    # per-query nDCG for every (norm, a): table[norm][a] = {q: ndcg}
    table = {}
    for method in NORMS:
        table[method] = {a: perq(combine(cands, scores, depth, a, method), set(cands)) for a in A_GRID}

    def macro(pq, qs):
        by = {}
        for q in qs:
            by.setdefault(qdom[q], []).append(pq[q])
        return float(np.mean([np.mean(v) for v in by.values()]))

    def fit(qs):
        """global (a, norm) and per-domain a* on the query subset qs."""
        g = max(((m, a) for m in NORMS for a in A_GRID),
                key=lambda x: (round(macro(table[x[0]][x[1]], qs), 6), -x[1]))
        per, n = {}, {}
        for d in DOMAINS:
            dq = [q for q in qs if qdom[q] == d]
            n[d] = len(dq)
            if dq:
                per[d] = float(max(A_GRID, key=lambda a: (round(np.mean([table[g[0]][a][q] for q in dq]), 6),
                                                   -abs(a - g[1]))))
        return g, per, n

    def shrink(g, per, n, m):
        return {d: float(g[1] if m == float("inf") else
                    round(min(A_GRID, key=lambda x: abs(x - (n[d] * per[d] + m * g[1]) / (n[d] + m))), 2))
                for d in per}

    def apply(g, ad, qs):
        pq = {q: table[g[0]][ad.get(qdom[q], g[1])][q] for q in qs}
        return macro(pq, qs)

    # 5-fold CV over queries, stratified by domain
    rng = random.Random(13)
    folds = [set() for _ in range(5)]
    for d in DOMAINS:
        dq = sorted(q for q in cands if qdom[q] == d)
        rng.shuffle(dq)
        for i, q in enumerate(dq):
            folds[i % 5].add(q)
    cv = {m: [] for m in M_GRID}
    for f in folds:
        tr = set(cands) - f
        g, per, n = fit(tr)
        for m in M_GRID:
            cv[m].append(apply(g, shrink(g, per, n, m), f))
    cvm = {m: float(np.mean(v)) for m, v in cv.items()}
    print("CV (5-fold within the tuning set):", {str(m): round(v, 4) for m, v in cvm.items()})
    best_m = max(M_GRID, key=lambda m: (round(cvm[m], 6), m))
    g, per, n = fit(set(cands))
    ad = shrink(g, per, n, best_m)
    print(f"global a={g[1]} {g[0]} -> {macro(table[g[0]][g[1]], set(cands)):.4f}; m={best_m}; "
          f"per-domain {ad} -> {apply(g, ad, set(cands)):.4f} (in-sample)")
    return {"a": float(g[1]), "norm": g[0], "a_per_domain": ad, "shrinkage_m": str(best_m),
            "cv": {str(m): round(v, 4) for m, v in cvm.items()}}


def tune_rrf3(cfg, qids, depth):
    cands = {q: v for q, v in train_candidates(cfg, 1000).items() if q in qids}
    scores, _, _ = score_candidates(cands, "train", dict(cfg["rerank"], seed=13), depth)
    shared = preload_shared(cfg)
    sp = shared[("sparse", tuple(DOMAINS))][0]["train"]
    dn = shared[("dense", tuple(DOMAINS))][0]["train"]
    rows = []
    grid_w = [round(x, 2) for x in np.arange(0, 1.0001, 0.1)]
    for k in [5, 10, 20, 40, 60, 100]:
        for wc in grid_w:
            for ws in grid_w:
                wd = round(1 - wc - ws, 2)
                if wd < -1e-9:
                    continue
                r = score_set(rrf3(cands, scores, depth, sp, dn, k, (ws, wd, wc)), set(cands), "t")
                rows.append({"k": k, "w": (ws, wd, wc), "ndcg": r["macro"]})
    best = max(rows, key=lambda r: round(r["ndcg"], 6))
    print(f"best rrf3 k={best['k']} w(sparse,dense,ce)={best['w']} -> {best['ndcg']:.4f} (in-sample)")
    top = sorted(rows, key=lambda r: -r["ndcg"])[:8]
    for r in top:
        print(f"   k={r['k']:<4} w={r['w']}  {r['ndcg']:.4f}")
    return {"method": "rrf3", "k": best["k"], "weights": [float(x) for x in best["w"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.fusion_exps")
    ap.add_argument("exp", choices=["per_domain", "rrf3"])
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--tune-on", default="tune", choices=["tune", "tune_val"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(load_yaml(args.config))
    qids = qset(args.tune_on)
    depth = cfg["rerank"]["depth"]
    ic = per_domain(cfg, qids, depth) if args.exp == "per_domain" else tune_rrf3(cfg, qids, depth)
    cfg["interpolate"] = ic
    cfg["system_name"] = "pg_" + args.exp
    with open(REPO_ROOT / args.out, "w") as f:
        f.write(f"# Playground {args.exp}, tuned on the {args.tune_on} part of train "
                f"(python -m reteco.playground.fusion_exps {args.exp} --tune-on {args.tune_on}).\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

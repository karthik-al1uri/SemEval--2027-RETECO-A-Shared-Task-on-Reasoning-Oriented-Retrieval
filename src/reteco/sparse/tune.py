"""Grid searches on TRAIN (dev is never read here).

    python -m reteco.sparse.tune bm25 --config configs/sparse_bm25_global.yaml
    python -m reteco.sparse.tune rm3  --config configs/sparse_rm3.yaml

`bm25` sweeps (k1, b) for the corpus / query cleaning given in --config and
reports the best global setting, the best per-domain settings, and a 5-fold
cross-validation (within train, folds over queries, seed 13) comparing the two
selection strategies on held-out train queries.

`rm3` keeps the config's (k1, b) (global or per-domain) and sweeps one global
(fb_docs, fb_terms, alpha).

Per-query nDCG@10 and Recall@100 for every grid point are cached per domain
under cache/sparse/tune/<config hash>/, so an interrupted sweep resumes.
Use --write to store the selected parameters back into the config (BM25:
k1/b and per_domain; RM3: the rm3 block).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytrec_eval
import yaml

from ..data import DOMAINS, load_qrels, load_queries
from .index import CACHE_DIR, TFIndex
from .retrieve import RM3, tokenize_queries
from .run import domain_params

SEED = 13
K1_GRID = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
B_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FB_DOCS = [3, 5, 10, 20]
FB_TERMS = [10, 20, 40, 80]
ALPHA = [0.3, 0.5, 0.7, 0.9]
METRICS = {"ndcg_cut.10", "recall.100"}


def _per_query(evaluator, qids, ranked, ids) -> np.ndarray:
    """(2, n_queries) array: nDCG@10, R@100 (queries with no results score 0)."""
    run = {q: {ids[r]: s for r, s in lst} for q, lst in zip(qids, ranked) if lst}
    res = evaluator.evaluate(run) if run else {}
    return np.array([[res.get(q, {}).get("ndcg_cut_10", 0.0) for q in qids],
                     [res.get(q, {}).get("recall_100", 0.0) for q in qids]])


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def _cache_dir(kind: str, config: dict) -> Path:
    base = {"corpus": config.get("corpus", "raw"), "query_clean": list(config.get("query_clean") or [])}
    if kind == "rm3":
        base["bm25"] = {d: domain_params(config, d) for d in DOMAINS}
        return CACHE_DIR / "tune" / f"rm3_{_key(base, [FB_DOCS, FB_TERMS, ALPHA])}"
    # BM25 results are cached per (k1, b), so the grid can be extended without recomputing.
    return CACHE_DIR / "tune" / f"bm25_{_key(base)}"


def _bm25_file(config: dict, domain: str, k1: float, b: float):
    return _cache_dir("bm25", config) / domain / f"k1-{k1}_b-{b}.npy"


def _setup(config: dict, domain: str):
    queries = load_queries(domain, "train")
    qrels = load_qrels(domain, "train")
    toks = tokenize_queries(queries, config.get("query_clean"))
    index = TFIndex.get(domain, config.get("corpus", "raw"))
    qids = [q.id for q in queries]
    return index, toks, qids, pytrec_eval.RelevanceEvaluator(qrels, METRICS)


def sweep_bm25(args) -> str:
    config, domain = args
    todo = [(k1, b) for k1 in K1_GRID for b in B_GRID if not _bm25_file(config, domain, k1, b).exists()]
    if not todo:
        return f"{domain} cached"
    t0 = time.time()
    index, toks, qids, ev = _setup(config, domain)
    view = index.restrict({t for q in toks for t in q})
    del index
    for k1, b in todo:
        w = view.bm25(k1, b)
        ranked = w.score_many([w.query_vector(t) for t in toks], 100)
        f = _bm25_file(config, domain, k1, b)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, _per_query(ev, qids, ranked, view.doc_ids))
    return f"{domain} {len(todo)} settings {time.time() - t0:.0f}s"


def sweep_rm3(args) -> str:
    config, domain = args
    out = _cache_dir("rm3", config) / f"{domain}.npz"
    if out.exists():
        return f"{domain} cached"
    t0 = time.time()
    index, toks, qids, ev = _setup(config, domain)
    k1, b = domain_params(config, domain)
    w = index.bm25(k1, b)
    model = RM3(index, (config.get("rm3") or {}).get("max_df", 0.1))
    qvecs = [w.query_vector(t) for t in toks]
    first = w.score_many(qvecs, depth=max(FB_DOCS))
    res = np.zeros((len(FB_DOCS), len(FB_TERMS), len(ALPHA), 2, len(qids)))
    for i, nd in enumerate(FB_DOCS):
        for j, nt in enumerate(FB_TERMS):
            fbs = [model.feedback_model(f, nd, nt) for f in first]
            for a, alpha in enumerate(ALPHA):
                qs = [model.interpolate(q, fb, alpha) if q else q for q, fb in zip(qvecs, fbs)]
                res[i, j, a] = _per_query(ev, qids, w.score_many(qs, 100), index.doc_ids)
    base = _per_query(ev, qids, w.score_many(qvecs, 100), index.doc_ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, res=res, base=base, qids=np.array(qids))
    return f"{domain} {time.time() - t0:.0f}s"


def _load(kind: str, config: dict) -> dict:
    if kind == "bm25":
        return {dom: {"res": np.array([[np.load(_bm25_file(config, dom, k1, b)) for b in B_GRID]
                                       for k1 in K1_GRID])} for dom in DOMAINS}
    d = _cache_dir(kind, config)
    return {dom: dict(np.load(d / f"{dom}.npz")) for dom in DOMAINS}


def _run_sweep(fn, config, workers):
    # largest domains first so the pool stays busy
    order = sorted(DOMAINS, key=lambda d: -len(load_queries(d, "train")))
    with Pool(workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(fn, [(config, d) for d in order]):
            print("  " + msg, flush=True)


def _cv(per_dom: dict[str, np.ndarray], folds: int = 5) -> dict:
    """5-fold CV over train queries. per_dom[d]: (n_settings, n_queries) nDCG.

    Returns held-out macro nDCG for global vs per-domain selection.
    """
    rng = np.random.RandomState(SEED)
    assign = {d: rng.permutation(np.arange(m.shape[1]) % folds) for d, m in per_dom.items()}
    held_glob = {d: np.zeros(m.shape[1]) for d, m in per_dom.items()}
    held_dom = {d: np.zeros(m.shape[1]) for d, m in per_dom.items()}
    for f in range(folds):
        train_means = {d: m[:, assign[d] != f].mean(axis=1) for d, m in per_dom.items()}
        g = int(np.argmax(np.mean([v for v in train_means.values()], axis=0)))
        for d, m in per_dom.items():
            te = assign[d] == f
            held_glob[d][te] = m[g, te]
            held_dom[d][te] = m[int(np.argmax(train_means[d])), te]
    return {"global": float(np.mean([v.mean() for v in held_glob.values()])),
            "per_domain": float(np.mean([v.mean() for v in held_dom.values()])),
            "per_domain_by_domain": {d: float(v.mean()) for d, v in held_dom.items()},
            "global_by_domain": {d: float(v.mean()) for d, v in held_glob.items()}}


def summarize_bm25(config: dict) -> dict:
    data = _load("bm25", config)
    settings = list(itertools.product(range(len(K1_GRID)), range(len(B_GRID))))
    flat = {d: v["res"].reshape(len(settings), 2, -1) for d, v in data.items()}
    ndcg = {d: m[:, 0, :] for d, m in flat.items()}
    rec = {d: m[:, 1, :] for d, m in flat.items()}
    macro_n = np.mean([m.mean(axis=1) for m in ndcg.values()], axis=0)
    macro_r = np.mean([m.mean(axis=1) for m in rec.values()], axis=0)
    g = int(np.argmax(macro_n))
    sp = lambda s: {"k1": K1_GRID[settings[s][0]], "b": B_GRID[settings[s][1]]}
    base = settings.index((K1_GRID.index(0.9), B_GRID.index(0.4)))
    per = {}
    for d in DOMAINS:
        s = int(np.argmax(ndcg[d].mean(axis=1)))
        per[d] = {**sp(s), "train_ndcg": float(ndcg[d][s].mean()), "train_recall": float(rec[d][s].mean()),
                  "global_ndcg": float(ndcg[d][g].mean()), "baseline_ndcg": float(ndcg[d][base].mean())}
    return {
        "grid": {"k1": K1_GRID, "b": B_GRID},
        "baseline": {**sp(base), "macro_ndcg": float(macro_n[base]), "macro_recall": float(macro_r[base])},
        "global": {**sp(g), "macro_ndcg": float(macro_n[g]), "macro_recall": float(macro_r[g])},
        "global_best_recall": {**sp(int(np.argmax(macro_r))), "macro_ndcg": float(macro_n[int(np.argmax(macro_r))]),
                               "macro_recall": float(macro_r.max())},
        "per_domain": per,
        "per_domain_macro_ndcg": float(np.mean([p["train_ndcg"] for p in per.values()])),
        "per_domain_macro_recall": float(np.mean([p["train_recall"] for p in per.values()])),
        "cv5": _cv(ndcg),
        "macro_ndcg_grid": macro_n.reshape(len(K1_GRID), len(B_GRID)).round(4).tolist(),
    }


def summarize_rm3(config: dict) -> dict:
    data = _load("rm3", config)
    shape = (len(FB_DOCS), len(FB_TERMS), len(ALPHA))
    settings = list(itertools.product(*[range(n) for n in shape]))
    flat = {d: v["res"].reshape(len(settings), 2, -1) for d, v in data.items()}
    macro_n = np.mean([m[:, 0].mean(axis=1) for m in flat.values()], axis=0)
    macro_r = np.mean([m[:, 1].mean(axis=1) for m in flat.values()], axis=0)
    base_n = float(np.mean([v["base"][0].mean() for v in data.values()]))
    base_r = float(np.mean([v["base"][1].mean() for v in data.values()]))
    sp = lambda s: {"fb_docs": FB_DOCS[settings[s][0]], "fb_terms": FB_TERMS[settings[s][1]],
                    "alpha": ALPHA[settings[s][2]]}
    g, gr = int(np.argmax(macro_n)), int(np.argmax(macro_r))
    top = np.argsort(-macro_n)[:10]
    return {
        "grid": {"fb_docs": FB_DOCS, "fb_terms": FB_TERMS, "alpha": ALPHA},
        "no_rm3": {"macro_ndcg": base_n, "macro_recall": base_r},
        "best": {**sp(g), "macro_ndcg": float(macro_n[g]), "macro_recall": float(macro_r[g]),
                 "by_domain": {d: float(m[g, 0].mean()) for d, m in flat.items()}},
        "best_recall": {**sp(gr), "macro_ndcg": float(macro_n[gr]), "macro_recall": float(macro_r[gr])},
        "top10": [{**sp(int(s)), "macro_ndcg": round(float(macro_n[s]), 4),
                   "macro_recall": round(float(macro_r[s]), 4)} for s in top],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.sparse.tune")
    ap.add_argument("kind", choices=["bm25", "rm3"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write", choices=["global", "per_domain", "rm3"],
                    help="write the selected parameters into --config")
    args = ap.parse_args(argv)
    config = yaml.safe_load(open(args.config))
    t0 = time.time()
    _run_sweep(sweep_bm25 if args.kind == "bm25" else sweep_rm3, config, args.workers)
    summary = summarize_bm25(config) if args.kind == "bm25" else summarize_rm3(config)
    out = _cache_dir(args.kind, config) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("macro_ndcg_grid",)}, indent=2))
    print(f"summary: {out}  ({time.time() - t0:.0f}s)")
    if args.write:
        if args.write == "global":
            config.update(k1=summary["global"]["k1"], b=summary["global"]["b"])
            config.pop("per_domain", None)
        elif args.write == "per_domain":
            config.update(k1=summary["global"]["k1"], b=summary["global"]["b"])
            config["per_domain"] = {d: {"k1": p["k1"], "b": p["b"]} for d, p in summary["per_domain"].items()}
        else:
            b = summary["best"]
            config["rm3"] = {"fb_docs": b["fb_docs"], "fb_terms": b["fb_terms"], "alpha": b["alpha"],
                             "max_df": (config.get("rm3") or {}).get("max_df", 0.1)}
        with open(args.config, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"wrote selected parameters to {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

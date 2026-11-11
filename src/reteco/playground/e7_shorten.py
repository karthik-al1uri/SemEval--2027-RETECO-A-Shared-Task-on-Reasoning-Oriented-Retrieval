"""E7: grid over extractive query-shortening variants, first stage only, on tune_val.

    python -m reteco.playground.e7_shorten --config configs/pg_e8.yaml

For every variant (first_n x time_sents x title_weight x stages) the sparse and dense
depth-1000 lists of the tune_val queries are recomputed, fused with the champion's RRF
setting and scored: hybrid nDCG@10 and Recall@200 (the reranker's input). Each domain's
index and embeddings are loaded once for all variants.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from ..data import DOMAINS, REPO_ROOT, load_queries
from ..dense.encode import encode_queries, load_domain, load_model
from ..dense.fusion import ranks, rrf
from ..dense.retrieve import search_first
from ..pipeline import load_yaml
from ..sparse.index import TFIndex
from ..sparse.retrieve import search, tokenize_queries
from ..sparse.run import domain_params
from .common import _qrels, score_set, subsplit_tune
from .qshorten import shorten_queries


def recall_at(run, qids, k=200):
    by = {}
    for d in DOMAINS:
        qr = {q: r for q, r in _qrels(d).items() if q in qids}
        vals = []
        for q, rel in qr.items():
            gold = {x for x, v in rel.items() if v > 0}
            top = sorted(run.get(q, {}).items(), key=lambda x: -x[1])[:k]
            vals.append(len(gold & {x for x, _ in top}) / max(1, len(gold)))
        by[d] = np.mean(vals)
    return float(np.mean(list(by.values())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.e7_shorten")
    ap.add_argument("--config", default="configs/pg_e8.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    scfg = load_yaml(cfg["sparse"]["config"])
    dcfg = load_yaml(cfg["dense"]["config"])
    _, val = subsplit_tune()
    variants = [None] + [
        {"first_n": n, "time_sents": t, "title_weight": w, "stages": st}
        for n, t, w, st in itertools.product([1, 2, 3, 5], [True, False], [1, 2],
                                             [("sparse",), ("sparse", "dense")])]
    variants.append({"first_n": None, "time_sents": False, "title_weight": 2, "stages": ("sparse",)})
    variants.append({"first_n": None, "time_sents": False, "title_weight": 2, "stages": ("sparse", "dense")})
    qm = dcfg.get("query_model")
    model = load_model(str(REPO_ROOT / qm) if qm else dcfg["model"], dcfg.get("dtype", "float16"))
    sp = {i: {} for i in range(len(variants))}
    dn = {i: {} for i in range(len(variants))}
    for dom in DOMAINS:
        qs = [q for q in load_queries(dom, "train") if q.id in val]
        index = TFIndex.get(dom, scfg.get("corpus", "raw"), 8)
        k1, b = domain_params(scfg, dom)
        weights = index.bm25(k1, b)
        emb, ids, _, _ = load_domain(dcfg, dom)
        cache_s, cache_d = {}, {}
        for i, v in enumerate(variants):
            stages = v["stages"] if v else ()
            ks = repr(v) if "sparse" in stages else "full"
            kd = repr(v) if "dense" in stages else "full"
            if ks not in cache_s:
                q2 = shorten_queries(qs, v) if "sparse" in stages else qs
                cache_s[ks] = [dict(r) for r in search(index, tokenize_queries(q2, scfg.get("query_clean")),
                                                       k1, b, 1000, None, weights=weights)]
            if kd not in cache_d:
                q2 = shorten_queries(qs, v) if "dense" in stages else qs
                cache_d[kd] = search_first(emb, ids, encode_queries(model, [x.query for x in q2], dcfg), 1000)
            for q, rs, rd in zip(qs, cache_s[ks], cache_d[kd]):
                sp[i][q.id] = rs
                dn[i][q.id] = rd
        del index, weights, emb
        print(f"{dom} done", flush=True)
    f = cfg["fusion"]
    rows = []
    for i, v in enumerate(variants):
        hyb = rrf([ranks(sp[i]), ranks(dn[i])], [1 - f["w_dense"], f["w_dense"]], f["k"], 1000)
        top100 = {q: dict(sorted(x.items(), key=lambda y: -y[1])[:100]) for q, x in hyb.items()}
        n = score_set(top100, val, "v")["macro"]
        r = recall_at(hyb, val, 200)
        rows.append((v, n, r))
        print(f"VAR {v}  hybrid nDCG@10 {n:.4f}  R@200 {r:.4f}", flush=True)
    best = max(rows, key=lambda x: round(x[1], 6))
    print(f"BEST {best[0]}  {best[1]:.4f}  R@200 {best[2]:.4f}  (baseline {rows[0][1]:.4f} / {rows[0][2]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

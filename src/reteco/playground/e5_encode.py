"""R2-E5: encode corpora with the fully fine-tuned bi-encoder (small domains first) and compare
first-stage retrieval on the small domains' tune_val queries against round-1 E8's query encoder
(trained on tune_train only, frozen document vectors), the fair out-of-sample reference.

    python -m reteco.playground.e5_encode --small            encode + evaluate small domains
    python -m reteco.playground.e5_encode --rest             encode the other 8 domains
"""
from __future__ import annotations

import argparse

from ..data import DOMAINS, REPO_ROOT, SMALL_DOMAINS, load_queries
from ..dense.encode import encode_domain, encode_queries, load_domain, load_model
from ..dense.fusion import ranks, rrf
from ..dense.retrieve import search_first
from ..pipeline import load_yaml
from .common import score_set, subsplit_tune
from .harness import preload_shared

NEW = "configs/pg2_e5_dense.yaml"
E8_QENC = "cache/playground/qenc/bge_q/epoch2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.e5_encode")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--rest", action="store_true")
    args = ap.parse_args(argv)
    ncfg = load_yaml(NEW)
    model = load_model(str(REPO_ROOT / ncfg["model"]), ncfg.get("dtype", "float16"))
    doms = (SMALL_DOMAINS if args.small else []) + ([d for d in DOMAINS if d not in SMALL_DOMAINS] if args.rest else [])
    for d in doms:
        encode_domain(ncfg, d, model)
    if not args.small:
        return 0
    cfg = load_yaml("configs/pg2_e3.yaml")
    ocfg = load_yaml(cfg["dense"]["config"])
    sparse = preload_shared(cfg)[("sparse", tuple(DOMAINS))][0]["train"]
    _, tv = subsplit_tune()
    f = cfg["fusion"]
    e8 = load_model(str(REPO_ROOT / E8_QENC), "float16")
    for name, qm, dc in (("E8 query encoder + cached docs", e8, ocfg), ("full bi-encoder + new docs", model, ncfg)):
        dense = {}
        for d in SMALL_DOMAINS:
            emb, ids, _, _ = load_domain(dc, d)
            qs = [q for q in load_queries(d, "train") if q.id in tv]
            for q, r in zip(qs, search_first(emb, ids, encode_queries(qm, [x.query for x in qs], dc), 1000)):
                dense[q.id] = r
        hyb = rrf([ranks({q: sparse[q] for q in dense}), ranks(dense)], [1 - f["w_dense"], f["w_dense"]], f["k"], 100)
        top = {q: dict(sorted(v.items(), key=lambda x: -x[1])[:100]) for q, v in dense.items()}
        a, b = score_set(top, tv, name, SMALL_DOMAINS), score_set(hyb, tv, name, SMALL_DOMAINS)
        print(f"SMALL {name}: dense {a['macro']:.4f} hybrid {b['macro']:.4f} R@100 {b['recall@100']:.4f}  "
              + " ".join(f"{k[:4]}={v:.3f}" for k, v in b["domains"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

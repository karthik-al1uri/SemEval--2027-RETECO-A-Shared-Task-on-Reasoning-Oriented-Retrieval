"""E6: second dense model (e5-small-v2) fused into the first stage, 5 small domains only.

    python -m reteco.playground.e6_second_dense --config configs/pg_e3.yaml

Full-corpus encoding would exceed the 2-hour cap, so this is a first-stage-only check on the
small domains (iota, quant, law, workplace, cardano): weighted RRF of sparse, bge (the
champion's dense, with its fine-tuned query encoder) and e5, with the champion's k. The e5
weight is tuned on the small-domain tune queries, (w_sparse : w_bge) kept at the champion's
ratio; scored on the small-domain holdout (27 queries).
"""
from __future__ import annotations

import argparse

import numpy as np

from ..data import SMALL_DOMAINS, load_queries
from ..dense.encode import encode_queries, load_domain, load_model
from ..dense.fusion import ranks, rrf
from ..dense.retrieve import search_first
from ..pipeline import load_yaml
from .common import part_qids, score_set
from .harness import preload_shared


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.e6_second_dense")
    ap.add_argument("--config", default="configs/pg_e3.yaml")
    ap.add_argument("--e5", default="configs/pg_e6_dense_e5.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    shared = preload_shared(cfg)
    sp = shared[("sparse", tuple(__import__("reteco.data", fromlist=["DOMAINS"]).DOMAINS))][0]["train"]
    bg = shared[("dense", tuple(__import__("reteco.data", fromlist=["DOMAINS"]).DOMAINS))][0]["train"]
    ecfg = load_yaml(args.e5)
    model = load_model(ecfg["model"], "float16")
    e5 = {}
    for d in SMALL_DOMAINS:
        qs = load_queries(d, "train")
        emb, ids, _, _ = load_domain(ecfg, d)
        for q, r in zip(qs, search_first(emb, ids, encode_queries(model, [x.query for x in qs], ecfg), 1000)):
            e5[q.id] = r
    qids = set(e5)
    sp = {q: sp[q] for q in qids}
    bg = {q: bg[q] for q in qids}
    f = cfg["fusion"]
    k, wd = f["k"], f["w_dense"]
    tune = part_qids("tune", SMALL_DOMAINS)
    hold = part_qids("holdout", SMALL_DOMAINS)

    def run(w_e5):
        ws, wb = (1 - wd) * (1 - w_e5), wd * (1 - w_e5)
        lists, weights = [ranks(sp), ranks(bg)], [ws, wb]
        if w_e5 > 0:
            lists.append(ranks(e5))
            weights.append(w_e5)
        return rrf(lists, weights, k, 100)

    e5_alone = score_set(e5, tune, "t", SMALL_DOMAINS)["macro"]
    rows = []
    for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        r = run(w)
        rows.append((w, score_set(r, tune, "t", SMALL_DOMAINS)["macro"],
                     score_set(r, hold, "h", SMALL_DOMAINS)["macro"]))
        print(f"w_e5={w:.1f}  small tune {rows[-1][1]:.4f}  small holdout {rows[-1][2]:.4f}")
    best = max(rows, key=lambda r: (round(r[1], 6), -r[0]))
    print(f"e5 alone (small tune) {e5_alone:.4f}; bge alone {score_set(bg, tune, 't', SMALL_DOMAINS)['macro']:.4f}")
    print(f"chosen on small tune: w_e5={best[0]}  holdout {best[2]:.4f} vs {rows[0][2]:.4f} without e5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

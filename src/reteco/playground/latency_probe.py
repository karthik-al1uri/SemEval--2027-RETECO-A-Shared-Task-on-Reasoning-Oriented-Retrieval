"""Measure cross-encoder latency (ms per query, top `depth`) on a few holdout queries, without
touching the score cache. Used before training a larger cross-encoder (round 2, R2-E1).

    python -m reteco.playground.latency_probe --models cross-encoder/ms-marco-MiniLM-L-6-v2 \
        cross-encoder/ms-marco-MiniLM-L-12-v2 --n 30
"""
from __future__ import annotations

import argparse
import random
import time

from ..data import load_queries
from ..pipeline import load_yaml
from ..rerank.cross_encoder import CrossEncoder, load_docs
from .common import part_qids
from .harness import train_candidates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.latency_probe")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    rcfg = dict(cfg["rerank"])
    depth = rcfg["depth"]
    cands = train_candidates(cfg, depth)
    qs = sorted(part_qids("holdout"))
    random.Random(13).shuffle(qs)
    qs = qs[:args.n]
    texts, docs = {}, {}
    for dom in sorted({q.split("/")[0] if "/" in q else None for q in qs} - {None}) or []:
        pass
    by_dom = {}
    for q in qs:
        by_dom.setdefault(cands[q][0][0].split("/", 1)[0], []).append(q)
    for dom, ql in by_dom.items():
        texts.update({x.id: x.query for x in load_queries(dom, "train") if x.id in set(ql)})
        docs.update(load_docs(dom, {d for q in ql for d, _ in cands[q]}, rcfg.get("doc_chars", 4000)))
    for m in args.models:
        enc = CrossEncoder(dict(rcfg, model=m, seed=13))
        enc.warmup()
        for r in range(args.rounds):
            t = 0.0
            for q in qs:
                query = enc.prepare_query(texts[q], rcfg)
                _, sec = enc.score(query, [docs[d] for d, _ in cands[q]])
                t += sec
            print(f"{m}  round {r + 1}: {1000 * t / len(qs):.0f} ms/query (top {depth}, {len(qs)} queries)",
                  flush=True)
        del enc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

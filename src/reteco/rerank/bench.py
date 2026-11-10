"""Cross-encoder throughput on this machine, on real (query, candidate) pairs.

    python -m reteco.rerank.bench --models BAAI/bge-reranker-base cross-encoder/ms-marco-MiniLM-L-6-v2

Takes the hybrid_best top 50 of 6 train queries from the small domains (seed 13),
times scoring per query for several batch sizes, and compares fp16 with fp32
logits (Pearson r, Spearman rho per query). Writes cache/rerank/bench.json.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from ..data import SMALL_DOMAINS, load_queries
from ..dense.encode import hardware
from . import CACHE_DIR
from . import candidates as C
from .cross_encoder import CrossEncoder, load_docs


def sample_pairs(n_queries: int = 6, depth: int = 50, seed: int = 13):
    cands = C.load("hybrid_best", "train", SMALL_DOMAINS)
    rng = random.Random(seed)
    qids = rng.sample(sorted(cands), n_queries)
    texts = {q.id: q.query for d in SMALL_DOMAINS for q in load_queries(d, "train")}
    out = []
    for q in qids:
        docs = [d for d, _ in cands[q][:depth]]
        dom = docs[0].split("/", 1)[0]
        dt = load_docs(dom, set(docs), 4000)
        out.append((texts[q], [dt[d] for d in docs]))
    return out


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.rerank.bench")
    ap.add_argument("--models", nargs="+",
                    default=["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    args = ap.parse_args(argv)
    pairs = sample_pairs()
    n = sum(len(d) for _, d in pairs)
    report = {"hardware": hardware(), "pairs": n, "queries": len(pairs), "results": []}
    print(report["hardware"])
    for model in args.models:
        ref = None
        for dtype in ["float32", "float16"]:
            for bs in args.batch_sizes if dtype == "float16" else [32]:
                enc = CrossEncoder({"model": model, "dtype": dtype, "batch_size": bs,
                                    "max_length": 512, "max_query_tokens": 256,
                                    "query_clean": ["code", "html", "urls", "signature"]})
                secs, logits = 0.0, []
                for q, docs in pairs:
                    s, t = enc.score(enc.prepare_query(q), docs)
                    secs += t
                    logits.append(s)
                row = {"model": model, "dtype": dtype, "batch_size": bs,
                       "pairs_per_s": round(n / secs, 1), "ms_per_query_at_50": round(1000 * secs / len(pairs), 1)}
                if dtype == "float32":
                    ref = logits
                else:
                    row["pearson_vs_fp32"] = round(float(np.mean(
                        [np.corrcoef(a, b)[0, 1] for a, b in zip(ref, logits)])), 4)
                    row["spearman_vs_fp32"] = round(float(np.mean(
                        [spearman(a, b) for a, b in zip(ref, logits)])), 4)
                    row["finite"] = bool(all(np.isfinite(x).all() for x in logits))
                report["results"].append(row)
                print(row, flush=True)
                del enc
    path = CACHE_DIR / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

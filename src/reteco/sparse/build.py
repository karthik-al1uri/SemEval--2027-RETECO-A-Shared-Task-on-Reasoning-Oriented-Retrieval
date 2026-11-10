"""Build (or load from cache) the raw-tf indexes for each corpus variant and domain.

    python -m reteco.sparse.build --corpus raw markup clean

Resumable: finished domains are cached under cache/sparse/tf/ and skipped.
"""
from __future__ import annotations

import argparse
import time

from ..data import DOMAINS
from .clean import CORPUS_VARIANTS
from .index import TFIndex


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.sparse.build")
    ap.add_argument("--corpus", nargs="+", default=["raw"], choices=CORPUS_VARIANTS)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    for corpus in args.corpus:
        for domain in args.domains:
            t0 = time.time()
            done = (TFIndex.cache_path(domain, corpus) / "meta.json").exists()
            if not done:
                index = TFIndex.get(domain, corpus, args.workers)
                print(f"{corpus:<7} {domain:<10} {index.n_docs:>8} docs  avgdl {index.avgdl:7.1f}  "
                      f"vocab {len(index.vocab):>8}  {time.time() - t0:6.1f}s", flush=True)
            else:
                print(f"{corpus:<7} {domain:<10} cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

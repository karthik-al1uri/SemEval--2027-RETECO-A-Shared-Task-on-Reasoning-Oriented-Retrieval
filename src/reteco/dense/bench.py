"""Hardware check and document-encoding throughput, with full-corpus time estimates.

    python -m reteco.dense.bench --models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5

The sample takes documents from the start of every domain's corpus in
proportion to domain size, so its length mix matches the full 1.17M-document
Track 1 corpus. Results go to cache/dense/bench.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from ..data import DOMAINS, iter_corpus
from . import CACHE_DIR
from .encode import encode, hardware, load_model
from .text import split_passages

DOC_COUNTS = {"bitcoin": 76706, "cardano": 47509, "economics": 74692, "genealogy": 136101,
              "history": 200252, "hsm": 169552, "iota": 9680, "law": 36110, "monero": 53955,
              "politics": 135207, "quant": 25040, "travel": 158771, "workplace": 43584}
TOTAL_DOCS = sum(DOC_COUNTS.values())


def sample_docs(n: int) -> list[str]:
    docs = []
    for d in DOMAINS:
        k = max(1, round(n * DOC_COUNTS[d] / TOTAL_DOCS))
        docs.extend(t for _, t in itertools.islice(iter_corpus(d), k))
    return docs


def throughput(model, docs, max_len, batch_size) -> float:
    import torch
    model.max_seq_length = max_len
    encode(model, docs[:2 * batch_size], batch_size)  # warm-up
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t = time.time()
    encode(model, docs, batch_size)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return len(docs) / (time.time() - t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.dense.bench")
    ap.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    hw = hardware()
    print(json.dumps(hw, indent=2))
    docs = sample_docs(args.n)
    print(f"sample: {len(docs)} docs")
    out = {"hardware": hw, "sample_docs": len(docs), "batch_size": args.batch_size,
           "total_docs": TOTAL_DOCS, "rows": []}

    for name in args.models:
        ref = None
        for dtype in ["float16", "float32"]:
            model = load_model(name, dtype)
            if ref is None:
                # Token-length profile of the sample (same WordPiece vocab for bge/e5).
                lens = np.array([len(x) for x in model.tokenizer(docs, add_special_tokens=False,
                                                                  verbose=False)["input_ids"]])
                _, owner = split_passages(docs, model.tokenizer, 254, 192)
                out["token_profile"] = {
                    "mean": float(lens.mean()), "median": float(np.median(lens)),
                    "p90": float(np.percentile(lens, 90)),
                    "share_over_510": float((lens > 510).mean()),
                    "passages_per_doc_256s192": len(owner) / len(docs)}
            for max_len in ([512, 256] if dtype == "float16" else [512]):
                rate = throughput(model, docs, max_len, args.batch_size)
                row = {"model": name, "dtype": dtype, "max_tokens": max_len,
                       "docs_per_s": round(rate, 1),
                       "full_corpus_hours": round(TOTAL_DOCS / rate / 3600, 2)}
                if max_len == 512:
                    model.max_seq_length = 512
                    e = encode(model, docs[:256], args.batch_size)
                    if ref is None:
                        ref = e
                    else:  # fp32 vs fp16 agreement
                        row["cos_vs_fp16"] = float((e * ref).sum(1).mean())
                out["rows"].append(row)
                print(row, flush=True)
            del model
    (CACHE_DIR).mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "bench.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"token profile: {out['token_profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cached time expressions for documents and queries.

Documents: cache/temporal/doc_times/<EXTRACTOR_VERSION>/<domain>.jsonl, one
line per document: {"id": ..., "t": [[start, end, kind], ...]}. The cache is
append-only and filled on demand: `get_doc_times` extracts only the ids that
are not cached yet, so any run file (BM25, dense, hybrid) can be rescored and
only its new candidates cost extraction time.

    python -m reteco.temporal.doctimes --runs runs/train/bm25.trec runs/dev/bm25.trec
    python -m reteco.temporal.doctimes --all --domains iota   # whole corpus
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from ..data import DOMAINS, REPO_ROOT, SPLITS, iter_corpus, load_queries
from ..runs import read_run
from .extract import EXTRACTOR_VERSION, extract

CACHE_DIR = REPO_ROOT / "cache" / "temporal"
DOC_CACHE = CACHE_DIR / "doc_times" / EXTRACTOR_VERSION

Interval = tuple[int, int, str]


def _extract_batch(batch: list[tuple[str, str]]) -> list[tuple[str, list]]:
    return [(i, [[t.start, t.end, t.kind] for t in extract(text)]) for i, text in batch]


def _load_cache(domain: str) -> dict[str, list]:
    path = DOC_CACHE / f"{domain}.jsonl"
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                out[rec["id"]] = rec["t"]
    return out


def get_doc_times(domain: str, doc_ids, workers: int = 8, root=None,
                  verbose: bool = True) -> dict[str, list]:
    """doc_id -> [[start, end, kind], ...] for the requested ids (None = all docs)."""
    cached = _load_cache(domain)
    want_all = doc_ids is None
    need = None if want_all else set(doc_ids) - cached.keys()
    if want_all or need:
        t0 = time.time()

        def batches(size=200):
            buf = []
            for i, text in iter_corpus(domain, root):
                if i in cached or (need is not None and i not in need):
                    continue
                buf.append((i, text))
                if len(buf) == size:
                    yield buf
                    buf = []
            if buf:
                yield buf

        DOC_CACHE.mkdir(parents=True, exist_ok=True)
        n = 0
        with Pool(workers) as pool, open(DOC_CACHE / f"{domain}.jsonl", "a", encoding="utf-8") as f:
            for res in pool.imap_unordered(_extract_batch, batches(), chunksize=1):
                for i, t in res:
                    cached[i] = t
                    f.write(json.dumps({"id": i, "t": t}, separators=(",", ":")) + "\n")
                n += len(res)
        if verbose:
            print(f"  {domain}: extracted times for {n} docs in {time.time() - t0:.1f}s", flush=True)
        if need:
            missing = need - cached.keys()
            if missing:
                raise KeyError(f"{len(missing)} doc ids not found in {domain} corpus, e.g. {next(iter(missing))}")
    if want_all:
        return cached
    return {i: cached[i] for i in doc_ids}


def query_domains(splits=SPLITS, root=None) -> dict[str, str]:
    """qid -> domain (query ids are unique across domains and splits)."""
    return {q.id: d for d in DOMAINS for s in splits for q in load_queries(d, s, root)}


def run_doc_ids_by_domain(run: dict[str, dict[str, float]]) -> dict[str, set[str]]:
    """Group candidate doc ids by domain using the '<domain>/...' doc id prefix."""
    out: dict[str, set[str]] = defaultdict(set)
    for docs in run.values():
        for d in docs:
            out[d.split("/", 1)[0]].add(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.temporal.doctimes")
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--all", action="store_true", help="extract for every document in the domains")
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    need: dict[str, set] = defaultdict(set)
    for p in args.runs:
        for dom, ids in run_doc_ids_by_domain(read_run(p)).items():
            need[dom] |= ids
    for dom in args.domains:
        if args.all:
            get_doc_times(dom, None, args.workers)
        elif need.get(dom):
            get_doc_times(dom, need[dom], args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

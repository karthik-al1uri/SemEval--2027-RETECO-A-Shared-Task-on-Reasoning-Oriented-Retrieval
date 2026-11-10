"""BM25 baseline matching the official RETECO one (upstream TEMPO `retrieval_bm25`).

Official: pyserini Lucene analyzer + gensim LuceneBM25Model(k1=0.9, b=0.4) +
SparseMatrixSimilarity (no normalisation). gensim applies the BM25 term weight
to the query as well as to the documents, so the score is

    score(q, d) = sum_t  w(t, q) * w(t, d)
    w(t, x)     = idf(t) * tf(t, x) / (tf(t, x) + k1 * (1 - b + b * |x| / avgdl))
    idf(t)      = log(N + 1) - log(df(t) + 0.5)

where |x| counts in-vocabulary tokens. This module reproduces that with scipy
sparse matrices and the pure-Python analyzer in `reteco.analysis` (no JDK).

    python -m reteco.bm25 --config configs/bm25.yaml --splits train dev
"""
from __future__ import annotations

import argparse
import json
from array import array
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

from .analysis import ANALYZER_VERSION, LuceneEnglishAnalyzer
from .data import DOMAINS, REPO_ROOT, SPLITS, iter_corpus, load_queries
from .runs import write_run

CACHE_DIR = REPO_ROOT / "cache" / "bm25"

_analyzer = None


def _count_terms(texts: list[str]) -> list[Counter]:
    global _analyzer
    if _analyzer is None:
        _analyzer = LuceneEnglishAnalyzer()
    return [Counter(_analyzer.analyze(t)) for t in texts]


def _chunks(it, size):
    buf_ids, buf_txt = [], []
    for doc_id, text in it:
        buf_ids.append(doc_id)
        buf_txt.append(text)
        if len(buf_txt) == size:
            yield buf_ids, buf_txt
            buf_ids, buf_txt = [], []
    if buf_txt:
        yield buf_ids, buf_txt


class BM25Index:
    def __init__(self, doc_ids, vocab, tf: sp.csr_matrix, k1: float, b: float):
        self.doc_ids = list(doc_ids)
        self.vocab = vocab  # term -> column
        self.k1, self.b = k1, b
        self.n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        self.avgdl = float(doc_len.sum()) / self.n_docs
        df = np.diff(tf.tocsc().indptr)
        self.idf = (np.log(self.n_docs + 1.0) - np.log(df + 0.5)).astype(np.float64)
        # Document weights w(t, d), stored column-major for fast per-term slicing.
        w = tf.astype(np.float64).tocsr()
        norm = k1 * (1.0 - b + b * doc_len / self.avgdl)
        rows = np.repeat(np.arange(self.n_docs), np.diff(w.indptr))
        w.data = self.idf[w.indices] * w.data / (w.data + norm[rows])
        self.weights = w.astype(np.float32).tocsc()

    # ------------------------------------------------------------- building --
    @classmethod
    def build(cls, domain: str, k1: float, b: float, workers: int = 8, root=None,
              chunk: int = 500) -> "BM25Index":
        doc_ids, vocab = [], {}
        indptr, indices, data = array("q", [0]), array("i"), array("f")
        with Pool(workers) as pool:
            batches = _chunks(iter_corpus(domain, root), chunk)
            id_batches = []

            def texts():
                for ids, txt in batches:
                    id_batches.append(ids)
                    yield txt

            for counts in pool.imap(_count_terms, texts(), chunksize=1):
                doc_ids.extend(id_batches.pop(0))
                for c in counts:
                    for term, n in c.items():
                        col = vocab.get(term)
                        if col is None:
                            col = vocab[term] = len(vocab)
                        indices.append(col)
                        data.append(n)
                    indptr.append(len(indices))
        tf = sp.csr_matrix((np.frombuffer(data, dtype=np.float32), np.frombuffer(indices, dtype=np.int32),
                            np.frombuffer(indptr, dtype=np.int64)), shape=(len(doc_ids), len(vocab)))
        return cls(doc_ids, vocab, tf, k1, b)

    @staticmethod
    def cache_path(domain: str, k1: float, b: float) -> Path:
        return CACHE_DIR / f"{ANALYZER_VERSION}_k1-{k1}_b-{b}" / domain

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "weights.npz", self.weights)
        np.save(path / "idf.npy", self.idf)
        (path / "doc_ids.json").write_text(json.dumps(self.doc_ids))
        (path / "vocab.json").write_text(json.dumps(self.vocab))
        (path / "meta.json").write_text(json.dumps(
            {"k1": self.k1, "b": self.b, "avgdl": self.avgdl, "n_docs": self.n_docs,
             "analyzer": ANALYZER_VERSION}))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        self = cls.__new__(cls)
        meta = json.loads((path / "meta.json").read_text())
        self.k1, self.b, self.avgdl, self.n_docs = meta["k1"], meta["b"], meta["avgdl"], meta["n_docs"]
        self.weights = sp.load_npz(path / "weights.npz").tocsc()
        self.idf = np.load(path / "idf.npy")
        self.doc_ids = json.loads((path / "doc_ids.json").read_text())
        self.vocab = json.loads((path / "vocab.json").read_text())
        return self

    @classmethod
    def get(cls, domain: str, k1: float, b: float, workers: int = 8, root=None) -> "BM25Index":
        path = cls.cache_path(domain, k1, b)
        if (path / "meta.json").exists() and root is None:
            return cls.load(path)
        index = cls.build(domain, k1, b, workers, root)
        if root is None:
            index.save(path)
        return index

    # ------------------------------------------------------------ searching --
    def query_weights(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        counts = Counter(t for t in tokens if t in self.vocab)
        if not counts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        cols = np.array([self.vocab[t] for t in counts], dtype=np.int64)
        qtf = np.array(list(counts.values()), dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * qtf.sum() / self.avgdl)
        return cols, (self.idf[cols] * qtf / (qtf + norm)).astype(np.float32)

    def score(self, tokens: list[str]) -> np.ndarray:
        cols, qw = self.query_weights(tokens)
        if len(cols) == 0:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols] @ qw, dtype=np.float32).ravel()

    def search(self, tokens: list[str], depth: int = 100) -> dict[str, float]:
        scores = self.score(tokens)
        k = min(depth, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        return {self.doc_ids[i]: float(scores[i]) for i in top}


def run(config: dict, splits, domains, workers: int, out_dir: Path = REPO_ROOT / "runs",
        config_path: str = "") -> dict[str, Path]:
    analyzer = LuceneEnglishAnalyzer()
    runs = {s: {} for s in splits}
    for domain in domains:
        t0 = time.time()
        index = BM25Index.get(domain, config["k1"], config["b"], workers)
        t1 = time.time()
        for split in splits:
            for q in load_queries(domain, split):
                runs[split][q.id] = index.search(analyzer.analyze(q.query), config["depth"])
        print(f"{domain:<10} {index.n_docs:>8} docs  index {t1 - t0:6.1f}s  "
              f"search {time.time() - t1:5.1f}s", flush=True)
        del index
    written = {}
    for split in splits:
        path = write_run(runs[split], out_dir / split / f"{config['system_name']}.trec",
                         tag=config["system_name"], depth=config["depth"])
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"description": config["description"], "config": config_path,
             "domains": list(domains)}, indent=2) + "\n")
        written[split] = path
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.bm25")
    ap.add_argument("--config", default="configs/bm25.yaml")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    config = yaml.safe_load(open(args.config))
    run(config, args.splits, args.domains, args.workers, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

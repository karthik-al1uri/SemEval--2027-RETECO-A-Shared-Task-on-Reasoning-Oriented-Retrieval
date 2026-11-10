"""Raw term-frequency index, so BM25 weights can be recomputed for any (k1, b).

The baseline (`reteco.bm25`) caches weights for one fixed (k1, b). For tuning we
keep the raw counts instead:

    tf       docs x vocab CSR matrix of term counts (float32)
    doc_len  in-vocabulary token count per document
    df       document frequency per term

Scoring follows the baseline exactly (gensim LuceneBM25Model, query weighted too):

    score(q, d) = sum_t  w(t, q) * w(t, d)
    w(t, x)     = idf(t) * tf(t, x) / (tf(t, x) + k1 * (1 - b + b * |x| / avgdl))
    idf(t)      = log(N + 1) - log(df(t) + 0.5)

so `TFIndex(...).bm25(0.9, 0.4)` reproduces `runs/*/bm25.trec`.

Indexes live under cache/sparse/tf/<analyzer>_<corpus>/<domain>/, where
<corpus> is "raw" (documents as given) or "clean" (see `reteco.sparse.clean`).
cache/bm25/ is never touched.
"""
from __future__ import annotations

import json
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..analysis import ANALYZER_VERSION, LuceneEnglishAnalyzer
from ..data import REPO_ROOT, iter_corpus
from .clean import CORPUS_VARIANTS, clean_document, clean_markup, repeated_sentences

CACHE_DIR = REPO_ROOT / "cache" / "sparse"

_analyzer = None
_corpus = "raw"
_repeated: frozenset[str] = frozenset()


def _init_worker(corpus: str, repeated: frozenset[str]) -> None:
    global _analyzer, _corpus, _repeated
    _analyzer = LuceneEnglishAnalyzer()
    _corpus, _repeated = corpus, repeated


def _count_terms(texts: list[str]) -> list[Counter]:
    if _corpus == "markup":
        texts = [clean_markup(t) for t in texts]
    elif _corpus == "clean":
        texts = [clean_document(t, _repeated) for t in texts]
    return [Counter(_analyzer.analyze(t)) for t in texts]


def _chunks(it, size):
    ids, txt = [], []
    for doc_id, text in it:
        ids.append(doc_id)
        txt.append(text)
        if len(txt) == size:
            yield ids, txt
            ids, txt = [], []
    if txt:
        yield ids, txt


class BM25Weights:
    """BM25 document weights for one (k1, b), column-major for fast per-term slicing."""

    def __init__(self, index: "TFIndex", k1: float, b: float):
        self.index, self.k1, self.b = index, k1, b
        tf = index.tf_csc
        norm = k1 * (1.0 - b + b * index.doc_len / index.avgdl)
        data = tf.data.astype(np.float64)
        cols = np.repeat(np.arange(tf.shape[1]), np.diff(tf.indptr))
        data = index.idf[cols] * data / (data + norm[tf.indices])
        self.matrix = sp.csc_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)

    def query_vector(self, tokens: list[str]) -> dict[int, float]:
        """Baseline query weighting: BM25-saturated query tf with |q| length normalisation."""
        vocab = self.index.vocab
        counts = Counter(vocab[t] for t in tokens if t in vocab)
        if not counts:
            return {}
        qlen = sum(counts.values())
        norm = self.k1 * (1.0 - self.b + self.b * qlen / self.index.avgdl)
        idf = self.index.idf
        return {c: float(idf[c] * n / (n + norm)) for c, n in counts.items()}

    def score_many(self, qvecs: list[dict[int, float]], depth: int = 100,
                   batch: int = 64) -> list[list[tuple[int, float]]]:
        """Top-`depth` (doc_row, score) per weighted query, best first."""
        out = []
        n_docs = self.index.n_docs
        k = min(depth, n_docs)
        for start in range(0, len(qvecs), batch):
            chunk = qvecs[start:start + batch]
            cols = sorted({c for q in chunk for c in q})
            if not cols:
                out.extend([[] for _ in chunk])
                continue
            pos = {c: i for i, c in enumerate(cols)}
            rows, cidx, vals = [], [], []
            for j, q in enumerate(chunk):
                for c, w in q.items():
                    rows.append(pos[c])
                    cidx.append(j)
                    vals.append(w)
            Q = sp.csr_matrix((vals, (rows, cidx)), shape=(len(cols), len(chunk)), dtype=np.float32)
            S = (self.matrix[:, cols] @ Q).toarray()  # n_docs x len(chunk)
            for j, q in enumerate(chunk):
                if not q:
                    out.append([])
                    continue
                s = S[:, j]
                top = np.argpartition(-s, k - 1)[:k]
                top = top[s[top] > 0]
                top = top[np.argsort(-s[top], kind="stable")]
                out.append([(int(i), float(s[i])) for i in top])
        return out


class TFIndex:
    def __init__(self, doc_ids: list[str], vocab: dict[str, int], tf: sp.csr_matrix):
        self.doc_ids = list(doc_ids)
        self.vocab = vocab
        self.tf = tf.tocsr()
        self.tf.sort_indices()
        self._tf_csc = None
        self.n_docs = self.tf.shape[0]
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel().astype(np.float64)
        self.avgdl = float(self.doc_len.sum()) / self.n_docs
        self.df = np.diff(self.tf_csc.indptr).astype(np.float64)
        self.idf = np.log(self.n_docs + 1.0) - np.log(self.df + 0.5)
        self._terms = None

    @property
    def tf_csc(self) -> sp.csc_matrix:
        if self._tf_csc is None:
            self._tf_csc = self.tf.tocsc()
            self._tf_csc.sort_indices()
        return self._tf_csc

    @property
    def terms(self) -> list[str]:
        if self._terms is None:
            self._terms = [None] * len(self.vocab)
            for t, c in self.vocab.items():
                self._terms[c] = t
        return self._terms

    def bm25(self, k1: float, b: float) -> BM25Weights:
        return BM25Weights(self, k1, b)

    def restrict(self, terms) -> "TFIndex":
        """A view over a subset of terms (same N, avgdl, doc lengths and idf), for fast grid search."""
        cols = sorted({self.vocab[t] for t in terms if t in self.vocab})
        v = TFIndex.__new__(TFIndex)
        v.doc_ids, v.n_docs, v.doc_len, v.avgdl = self.doc_ids, self.n_docs, self.doc_len, self.avgdl
        v.vocab = {self.terms[c]: i for i, c in enumerate(cols)}
        v.tf = None
        v._tf_csc = self.tf_csc[:, cols]
        v.df, v.idf, v._terms = self.df[cols], self.idf[cols], None
        return v

    # ------------------------------------------------------------- building --
    @classmethod
    def build(cls, domain: str, corpus: str = "raw", workers: int = 8, chunk: int = 500) -> "TFIndex":
        if corpus not in CORPUS_VARIANTS:
            raise ValueError(f"unknown corpus variant {corpus!r}")
        doc_ids, vocab = [], {}
        indptr, indices, data = array("q", [0]), array("i"), array("f")
        id_batches = []

        def jobs():
            for ids, txt in _chunks(iter_corpus(domain), chunk):
                id_batches.append(ids)
                yield txt

        repeated = frozenset(repeated_sentences(domain, workers)) if corpus == "clean" else frozenset()
        with Pool(workers, initializer=_init_worker, initargs=(corpus, repeated)) as pool:
            for counts in pool.imap(_count_terms, jobs(), chunksize=1):
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
        return cls(doc_ids, vocab, tf)

    @staticmethod
    def cache_path(domain: str, corpus: str = "raw") -> Path:
        return CACHE_DIR / "tf" / f"{ANALYZER_VERSION}_{corpus}" / domain

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "tf.npz", self.tf)
        (path / "doc_ids.json").write_text(json.dumps(self.doc_ids))
        (path / "vocab.json").write_text(json.dumps(self.vocab))
        (path / "meta.json").write_text(json.dumps(
            {"n_docs": self.n_docs, "avgdl": self.avgdl, "analyzer": ANALYZER_VERSION}))

    @classmethod
    def load(cls, path: Path) -> "TFIndex":
        tf = sp.load_npz(path / "tf.npz")
        doc_ids = json.loads((path / "doc_ids.json").read_text())
        vocab = json.loads((path / "vocab.json").read_text())
        return cls(doc_ids, vocab, tf)

    @classmethod
    def get(cls, domain: str, corpus: str = "raw", workers: int = 8) -> "TFIndex":
        path = cls.cache_path(domain, corpus)
        if (path / "meta.json").exists():
            return cls.load(path)
        index = cls.build(domain, corpus, workers)
        index.save(path)
        return index

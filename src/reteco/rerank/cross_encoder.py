"""Cross-encoder scoring with a per-pair cache and per-call timing.

A cross-encoder reads (query, document) jointly and outputs one relevance
logit. The raw logit is stored (no sigmoid), so per-query normalisation in
`interpolate` sees the full range.

Input text:
  query     cleaned like sparse_best (reteco.sparse.clean.clean_query with the
            configured steps), then cut to `max_query_tokens` tokens
  document  first `doc_chars` characters (a cheap pre-cut; the tokenizer then
            fills the remaining budget up to `max_length` tokens)

Cache: cache/rerank/scores/<model key>/<split>/<domain>.jsonl, one line per
scoring call: {"qid", "seconds", "scores": {docid: logit}}. "seconds" is the
wall time of that call (tokenisation + forward pass + device sync) and is
spread evenly over its pairs, so the latency of any (query, depth) is the sum
of its pairs' shares. Calls are made per query in rank segments
(SEGMENTS = 0-25, 25-50, 50-100, 100-200, ...), so the latency of depth 50,
100 or 200 is a sum of whole calls, i.e. exactly what was measured.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import REPO_ROOT, data_dir, load_queries
from ..sparse.clean import clean_query
from . import CACHE_DIR

SEGMENTS = [0, 25, 50, 100, 200, 300, 500, 1000]
_ID = re.compile(r'"id"\s*:\s*"([^"]+)"')


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_key(cfg: dict) -> str:
    """Cache key: everything that changes a pair's score."""
    spec = {"model": cfg["model"], "dtype": cfg.get("dtype", "float16"),
            "max_length": cfg.get("max_length", 512),
            "max_query_tokens": cfg.get("max_query_tokens", 256),
            "doc_chars": cfg.get("doc_chars", 4000),
            "query_clean": list(cfg.get("query_clean") or [])}
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg['model'].replace('/', '-')}__{h}"


def load_docs(domain: str, ids: set[str], doc_chars: int, root=None) -> dict[str, str]:
    """Text of the given documents only (the corpus is streamed, ids matched before parsing)."""
    out = {}
    with open(data_dir(root) / domain / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            m = _ID.search(line, 0, 300)
            if m and m.group(1) in ids:
                rec = json.loads(line)
                out[str(rec["id"])] = rec["content"][:doc_chars]
    missing = ids - set(out)
    if missing:
        raise KeyError(f"{len(missing)} documents not found in {domain}, e.g. {next(iter(missing))}")
    return out


class CrossEncoder:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(cfg.get("seed", 13))
        self.torch = torch
        self.cfg = cfg
        self.device = device()
        # a relative path to a local (e.g. fine-tuned) model is resolved against the repo root
        local = REPO_ROOT / cfg["model"]
        src = str(local) if local.is_dir() else cfg["model"]
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            src, dtype=getattr(torch, cfg.get("dtype", "float16"))).to(self.device).eval()
        self.max_length = cfg.get("max_length", 512)
        self.batch_size = cfg.get("batch_size", 32)
        self._warm = False

    def prepare_query(self, text: str, cfg: dict | None = None) -> str:
        """Clean and cut a query with the settings of `cfg` (default: the encoder's own config).

        Query settings come from the caller's config because one loaded encoder may serve
        several configs that differ only in query handling (see `compatible`)."""
        cfg = cfg if cfg is not None else self.cfg
        steps = cfg.get("query_clean") or []
        q = clean_query(text, steps) if steps else text
        ids = self.tok(q, add_special_tokens=False, truncation=True,
                       max_length=cfg.get("max_query_tokens", 256))["input_ids"]
        return self.tok.decode(ids)

    def compatible(self, cfg: dict) -> bool:
        """Can this loaded model score pairs for `cfg`? (Same weights, precision and length.)"""
        return all(self.cfg.get(k, d) == cfg.get(k, d) for k, d in
                   (("model", None), ("dtype", "float16"), ("max_length", 512), ("batch_size", 32)))

    def _forward(self, query: str, docs: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                chunk = docs[i:i + self.batch_size]
                enc = self.tok([query] * len(chunk), chunk, padding=True, truncation="only_second",
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits[:, 0]
                out.append(logits.float().cpu().numpy())  # .cpu() waits for the device
        return np.concatenate(out) if out else np.zeros(0)

    def warmup(self):
        if not self._warm:
            self._forward("warm up query " * 20, ["warm up document " * 200] * self.batch_size)
            self._warm = True

    def score(self, query: str, docs: list[str]) -> tuple[np.ndarray, float]:
        """Logits for (query, doc) pairs and the wall time of the call in seconds."""
        self.warmup()
        t0 = time.perf_counter()
        s = self._forward(query, docs)
        return s, time.perf_counter() - t0


class ScoreCache:
    """Per (model key, split, domain) jsonl of scoring calls."""

    def __init__(self, cfg: dict, split: str):
        self.dir = CACHE_DIR / "scores" / model_key(cfg) / split
        self.dir.mkdir(parents=True, exist_ok=True)
        spec = self.dir.parent / "spec.json"
        if not spec.exists():
            spec.write_text(json.dumps({k: cfg.get(k) for k in
                                        ("model", "dtype", "max_length", "max_query_tokens",
                                         "doc_chars", "query_clean", "batch_size")}, indent=2) + "\n")
        self.scores: dict[str, dict[str, float]] = defaultdict(dict)
        self.secs: dict[str, dict[str, float]] = defaultdict(dict)  # per-pair share of call time
        self._loaded: set[str] = set()

    def path(self, domain: str) -> Path:
        return self.dir / f"{domain}.jsonl"

    def load(self, domain: str):
        if domain in self._loaded:
            return
        self._loaded.add(domain)
        p = self.path(domain)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # a line cut off by an interruption
                    continue
                share = rec["seconds"] / max(1, len(rec["scores"]))
                for d, s in rec["scores"].items():
                    self.scores[rec["qid"]][d] = s
                    self.secs[rec["qid"]][d] = share

    def add(self, domain: str, qid: str, docids: list[str], scores: np.ndarray, seconds: float):
        rec = {"qid": qid, "seconds": round(seconds, 5),
               "scores": {d: round(float(s), 5) for d, s in zip(docids, scores)}}
        with open(self.path(domain), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        share = seconds / max(1, len(docids))
        for d, s in rec["scores"].items():
            self.scores[qid][d] = s
            self.secs[qid][d] = share


def domain_of(qid_docs: list[tuple[str, float]]) -> str:
    return qid_docs[0][0].split("/", 1)[0]


def score_candidates(cands: dict[str, list[tuple[str, float]]], split: str, cfg: dict, depth: int,
                     encoder: CrossEncoder | None = None, root=None, log=print):
    """Cross-encoder logits for the top `depth` of every candidate list (cached).

    Returns (scores, seconds, encoder): qid -> {docid: logit}, qid -> measured
    seconds for scoring that query's top `depth` (sum of its pairs' shares), and
    the loaded encoder (None if everything came from the cache) for reuse.
    """
    cache = ScoreCache(cfg, split)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for q, docs in cands.items():
        if docs:
            by_domain[domain_of(docs)].append(q)
    for domain in sorted(by_domain):
        cache.load(domain)
        qids = by_domain[domain]
        todo = {q: [d for d, _ in cands[q][:depth] if d not in cache.scores[q]] for q in qids}
        n_todo = sum(map(len, todo.values()))
        if n_todo == 0:
            continue
        if encoder is None or not encoder.compatible(cfg):
            encoder = CrossEncoder(cfg)
        t0 = time.time()
        texts = {q.id: q.query for q in load_queries(domain, split, root)}
        doc_text = load_docs(domain, {d for v in todo.values() for d in v}, cfg.get("doc_chars", 4000), root)
        log(f"  {domain:<10} {len(qids)} queries, {n_todo} new pairs (docs loaded in {time.time() - t0:.0f}s)")
        t0, done = time.time(), 0
        for q in qids:
            if not todo[q]:
                continue
            query = encoder.prepare_query(texts[q], cfg)
            pos = {d: r for r, (d, _) in enumerate(cands[q])}
            for lo, hi in zip(SEGMENTS, SEGMENTS[1:]):
                seg = [d for d in todo[q] if lo <= pos[d] < hi]
                if seg:
                    s, sec = encoder.score(query, [doc_text[d] for d in seg])
                    cache.add(domain, q, seg, s, sec)
                    done += len(seg)
        el = time.time() - t0
        log(f"  {domain:<10} scored {done} pairs in {el:.0f}s ({done / max(el, 1e-9):.1f} pairs/s)")
    scores, seconds = {}, {}
    for q, docs in cands.items():
        top = [d for d, _ in docs[:depth]]
        scores[q] = {d: cache.scores[q][d] for d in top}
        seconds[q] = sum(cache.secs[q][d] for d in top)
    return scores, seconds, encoder

"""Loaders for RETECO Track 1 (TEMPO) data, v1.1.

Layout: <data_dir>/<domain>/{documents.jsonl, examples_<split>.jsonl, qrels_<split>.txt, ...}

`load_queries` returns only `id` and `query`, which is all a system may see at
prediction time. Gold labels are only reachable through `load_qrels`, which is
meant for evaluation and for training labels on the train split.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "track1_tempo"

DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]
SMALL_DOMAINS = ["iota", "quant", "law", "workplace", "cardano"]
SPLITS = ["train", "dev"]


@dataclass(frozen=True)
class Query:
    id: str
    query: str


def data_dir(path: str | os.PathLike | None = None) -> Path:
    """Resolve the Track 1 data directory: explicit path, $RETECO_DATA, or data/track1_tempo."""
    if path is not None:
        return Path(path)
    env = os.environ.get("RETECO_DATA")
    return Path(env) if env else DEFAULT_DATA_DIR


def _check(domain: str, split: str | None = None) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def load_queries(domain: str, split: str, root=None) -> list[Query]:
    """Queries for one domain and split, reduced to (id, query)."""
    _check(domain, split)
    out = []
    with open(data_dir(root) / domain / f"examples_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out.append(Query(id=str(rec["id"]), query=rec["query"]))
    return out


def iter_corpus(domain: str, root=None) -> Iterator[tuple[str, str]]:
    """Stream (doc_id, content) pairs from a domain's documents.jsonl."""
    _check(domain)
    with open(data_dir(root) / domain / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield str(rec["id"]), rec["content"]


def load_qrels(domain: str, split: str, root=None) -> dict[str, dict[str, int]]:
    """qid -> {doc_id: relevance} from qrels_<split>.txt (TREC format: qid 0 docid rel)."""
    _check(domain, split)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(data_dir(root) / domain / f"qrels_{split}.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels[qid][docid] = int(rel)
    return dict(qrels)


def load_all_qrels(split: str, domains=None, root=None) -> dict[str, dict[str, dict[str, int]]]:
    """domain -> qrels for the given split."""
    return {d: load_qrels(d, split, root) for d in (domains or DOMAINS)}

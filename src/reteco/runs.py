"""TREC run files: `qid Q0 docid rank score tag`, one line per retrieved document.

Lines are written in trec_eval order (score descending, ties broken by docid
descending, after rounding the score to the precision written to disk), so the
rank column and the score column always give the same ordering.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

DEFAULT_DEPTH = 100
SCORE_DECIMALS = 6


def _ordered(docs: dict[str, float], depth: int) -> list[tuple[str, float]]:
    rounded = [(d, round(float(s), SCORE_DECIMALS)) for d, s in docs.items()]
    # Two stable sorts: docid descending, then score descending.
    rounded.sort(key=lambda x: x[0], reverse=True)
    rounded.sort(key=lambda x: x[1], reverse=True)
    return rounded[:depth]


def write_run(run: dict[str, dict[str, float]], path: str | os.PathLike,
              tag: str, depth: int = DEFAULT_DEPTH) -> Path:
    """Write qid -> {docid: score} as a TREC run, keeping the top `depth` per query."""
    if any(c.isspace() for c in tag) or not tag:
        raise ValueError("tag must be a non-empty string without whitespace")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(run):
            for rank, (docid, score) in enumerate(_ordered(run[qid], depth), 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.{SCORE_DECIMALS}f} {tag}\n")
    return path


def read_run(path: str | os.PathLike) -> dict[str, dict[str, float]]:
    """Read a TREC run into qid -> {docid: score}. Raises on malformed lines or duplicates."""
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"{path}:{n}: expected 6 columns, got {len(parts)}")
            qid, _q0, docid, _rank, score, _tag = parts
            if docid in run[qid]:
                raise ValueError(f"{path}:{n}: duplicate document {docid} for query {qid}")
            run[qid][docid] = float(score)
    return dict(run)

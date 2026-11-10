"""Helpers for error analysis: fetch documents by id, rank lookups, simple signals.

Gold labels are read here only to describe failures of an existing run. Nothing
computed here feeds back into a system.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from .. import SEED
from ..data import data_dir
from ..temporal.extract import clean_text, extract

_TOKEN = re.compile(r"[a-z][a-z0-9']+")
STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be
been being this that these those it its i me my we our you your he she they them their his her
what which who whom whose when where why how do does did done have has had not no yes can could
would should will shall may might must about into over under than so such there here also just
any all some more most other only very much many one two get got like know think want use used
using way even still out up down off again further once same each both few own too s t don""".split())


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(clean_text(text).lower()) if t not in STOP and len(t) > 2}


def fetch_docs(ids: set[str], root=None) -> dict[str, str]:
    """doc id -> content, streaming only the domains the ids belong to."""
    by_dom = defaultdict(set)
    for i in ids:
        by_dom[i.split("/", 1)[0]].add(i)
    out = {}
    prefix = '{"id": "'
    for dom, want in by_dom.items():
        with open(data_dir(root) / dom / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.startswith(prefix):
                    did = line[len(prefix):line.index('"', len(prefix))]
                    if did not in want:
                        continue
                rec = json.loads(line)
                if rec["id"] in want:
                    out[rec["id"]] = rec["content"]
                    if len(out) == len(ids):
                        return out
    return out


def ranks(run: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    """qid -> {docid: 1-based rank}, ranking by score desc, docid desc (trec_eval order)."""
    out = {}
    for q, docs in run.items():
        order = sorted(docs.items(), key=lambda x: x[0], reverse=True)
        order.sort(key=lambda x: x[1], reverse=True)
        out[q] = {d: r for r, (d, _) in enumerate(order, 1)}
    return out


def years(text: str, present: bool = False) -> list[tuple[int, int]]:
    return [(e.start, e.end) for e in extract(clean_text(text), present=present)
            if e.kind != "present" or present]


def doc_prox(q_iv, d_iv, tau: float = 25.0, max_width: int = 500) -> float | None:
    """Share of a document's time mentions near the query's time (temporal-fit doc_prox)."""
    q_iv = [i for i in q_iv if i[1] - i[0] <= max_width]
    d_iv = [i for i in d_iv if i[1] - i[0] <= max_width]
    if not q_iv or not d_iv:
        return None
    vals = []
    for ds, de in d_iv:
        best = 0.0
        for qs, qe in q_iv:
            dist = max(0, ds - qe, qs - de)
            best = max(best, float(np.exp(-dist / tau)))
        vals.append(best)
    return float(np.mean(vals))


def stratified_sample(items_by_domain: dict[str, list[str]], k: int, seed: int = SEED) -> list[str]:
    """k items, at least one per domain (while k allows), rest proportional (largest remainder)."""
    rng = np.random.default_rng(seed)
    doms = sorted(d for d, v in items_by_domain.items() if v)
    alloc = {d: 1 for d in doms} if k >= len(doms) else {}
    left = k - sum(alloc.values())
    rest = {d: len(items_by_domain[d]) - alloc.get(d, 0) for d in doms}
    rest_total = sum(rest.values())
    quota = {d: left * rest[d] / rest_total for d in doms}
    for d in doms:
        alloc[d] = alloc.get(d, 0) + int(np.floor(quota[d]))
    rem = k - sum(alloc.values())
    for d in sorted(doms, key=lambda d: (-(quota[d] - np.floor(quota[d])), d))[:rem]:
        alloc[d] += 1
    out = []
    for d in doms:
        pool = sorted(items_by_domain[d])
        n = min(alloc[d], len(pool))
        out += [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return out

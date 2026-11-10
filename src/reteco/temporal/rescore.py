"""Temporal-fit rescorer for any TREC run file.

    final(q, d) = a * norm(retrieval(q, d)) + b * g(q) * temporal(q, d)

norm() is a per-query normalisation of the input run's scores (min-max by
default, so BM25, dense and hybrid runs live on the same [0, 1] scale), and
g(q) is an optional gate in [0, 1] (1 when no gate is configured; a
classifier probability, or 0/1 above a threshold, otherwise). Queries or
documents without time expressions get temporal = 0, so their order is
unchanged.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo \\
    python -m reteco.temporal.rescore --run runs/dev/bm25.trec --split dev \\
        --config configs/temporal_fit.yaml

Writes runs/<split>/<input stem>+<system_name>.trec (or --out) and appends
the score to results/experiments.csv unless --no-log.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from ..data import REPO_ROOT, SPLITS, load_qrels, load_queries
from ..evaluate import append_result, evaluate_run, format_table
from ..runs import read_run, write_run
from .doctimes import get_doc_times, run_doc_ids_by_domain
from .extract import extract
from .fit import FitParams, PairFeatures


@dataclass
class Table:
    """A run flattened to aligned arrays (one row per (query, doc) candidate)."""
    qids: list[str]          # per query
    domains: list[str]       # per query
    offsets: np.ndarray      # query i owns rows offsets[i]:offsets[i+1]
    docids: list[str]        # per row
    retrieval: np.ndarray    # per row, raw input score
    q_times: list[list]      # per query
    d_times: list[list]      # per row

    @property
    def row_query(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.qids)), np.diff(self.offsets))


def query_times(text: str, present: bool = False) -> list:
    return [[t.start, t.end, t.kind] for t in extract(text, present=present)]


def build_table(run: dict[str, dict[str, float]], split: str, depth: int = 100,
                present: bool = False, workers: int = 8, root=None) -> Table:
    texts, q_domain = {}, {}
    run_domains = sorted({doc.split("/", 1)[0] for docs in run.values() for doc in docs})
    for d in run_domains:
        for q in load_queries(d, split, root):
            texts[q.id] = q.query
            q_domain[q.id] = d
    unknown = [q for q in run if q not in texts]
    if unknown:
        raise ValueError(f"{len(unknown)} run queries are not {split} queries, e.g. {unknown[0]}")
    # keep the top `depth` of each query
    run = {q: dict(sorted(docs.items(), key=lambda x: -x[1])[:depth]) for q, docs in run.items()}
    dtimes: dict[str, list] = {}
    for dom, ids in sorted(run_doc_ids_by_domain(run).items()):
        dtimes.update(get_doc_times(dom, ids, workers, root))
    qids = sorted(run)
    offsets, docids, scores, d_times = [0], [], [], []
    for q in qids:
        for doc, s in run[q].items():
            docids.append(doc)
            scores.append(s)
            d_times.append(dtimes[doc])
        offsets.append(len(docids))
    q_times = [query_times(texts[q], present) for q in qids]
    return Table(qids, [q_domain[q] for q in qids], np.asarray(offsets), docids,
                 np.asarray(scores, dtype=np.float64), q_times, d_times)


def normalize(table: Table, method: str = "minmax") -> np.ndarray:
    out = np.empty_like(table.retrieval)
    for i in range(len(table.qids)):
        a, b = table.offsets[i], table.offsets[i + 1]
        x = table.retrieval[a:b]
        if method == "minmax":
            rng = x.max() - x.min()
            out[a:b] = (x - x.min()) / rng if rng > 0 else 0.0
        elif method == "zscore":
            sd = x.std()
            out[a:b] = (x - x.mean()) / sd if sd > 0 else 0.0
        elif method == "max":
            out[a:b] = x / x.max() if x.max() > 0 else 0.0
        else:
            raise ValueError(f"unknown normalisation {method!r}")
    return out


def pair_features(table: Table, p: FitParams) -> PairFeatures:
    rq = table.row_query
    return PairFeatures([table.q_times[i] for i in rq], table.d_times, p)


def combine(table: Table, norm: np.ndarray, temporal: np.ndarray, a: float, b: float,
            gate: np.ndarray | None = None) -> np.ndarray:
    g = 1.0 if gate is None else gate[table.row_query]
    return a * norm + b * g * temporal


def to_run(table: Table, scores: np.ndarray) -> dict[str, dict[str, float]]:
    run = {}
    for i, q in enumerate(table.qids):
        a, b = table.offsets[i], table.offsets[i + 1]
        run[q] = dict(zip(table.docids[a:b], scores[a:b].tolist()))
    return run


# ---------------------------------------------------------------- fast eval --
class FastNDCG:
    """nDCG@10 per query, matching pytrec_eval / trec_eval conventions.

    Gains are the qrels relevance values; the ideal DCG uses every judged
    document (also unretrieved ones); ties are broken by doc id descending
    after rounding scores to 6 decimals, like trec_eval on our run files.
    """

    def __init__(self, table: Table, split: str, root=None, k: int = 10):
        self.k = k
        self.table = table
        qrels = {}
        for d in set(table.domains):
            qrels.update(load_qrels(d, split, root))
        self.rel = np.zeros(len(table.docids))
        self.idcg = np.zeros(len(table.qids))
        for i, q in enumerate(table.qids):
            a, b = table.offsets[i], table.offsets[i + 1]
            qr = qrels.get(q, {})
            self.rel[a:b] = [qr.get(d, 0) for d in table.docids[a:b]]
            ideal = sorted((r for r in qr.values() if r > 0), reverse=True)[:k]
            self.idcg[i] = sum(r / np.log2(j + 2) for j, r in enumerate(ideal))
        # doc-id rank within each query for the tie-break (higher id first)
        self.id_rank = np.zeros(len(table.docids))
        for i in range(len(table.qids)):
            a, b = table.offsets[i], table.offsets[i + 1]
            order = sorted(range(a, b), key=lambda j: table.docids[j])
            self.id_rank[order] = np.arange(b - a)
        self.disc = 1.0 / np.log2(np.arange(2, k + 2))
        # judged queries that are absent from the run still count (score 0)
        self.domain_n = {d: len(load_qrels(d, split, root)) for d in set(table.domains)}

    def per_query(self, scores: np.ndarray) -> np.ndarray:
        s = np.round(scores, 6)
        out = np.zeros(len(self.table.qids))
        off = self.table.offsets
        for i in range(len(out)):
            if self.idcg[i] == 0:
                continue
            a, b = off[i], off[i + 1]
            order = np.lexsort((-self.id_rank[a:b], -s[a:b]))[: self.k]
            out[i] = (self.rel[a:b][order] * self.disc[: len(order)]).sum() / self.idcg[i]
        return out

    def per_domain(self, scores: np.ndarray) -> dict[str, float]:
        pq = self.per_query(scores)
        dom = np.asarray(self.table.domains)
        return {d: float(pq[dom == d].sum() / self.domain_n[d]) for d in sorted(self.domain_n)}

    def macro(self, scores: np.ndarray) -> float:
        return float(np.mean(list(self.per_domain(scores).values())))


# ------------------------------------------------------------------- config --
def fit_params(cfg: dict) -> FitParams:
    names = {f.name for f in fields(FitParams)}
    kw = {k: v for k, v in cfg.get("fit", {}).items() if k in names}
    if "drop_kinds" in kw:
        kw["drop_kinds"] = tuple(kw["drop_kinds"])
    return FitParams(**kw)


def load_gate(cfg: dict, table: Table, split: str) -> np.ndarray | None:
    gcfg = cfg.get("gate")
    if not gcfg:
        return None
    from .classify import predict_gate

    probs = predict_gate(table, split, gcfg)
    thr = gcfg.get("threshold")
    return (probs >= thr).astype(float) if thr is not None else probs


def rescore_run(run: dict, split: str, cfg: dict, workers: int = 8, root=None):
    depth = cfg.get("depth", 100)
    table = build_table(run, split, depth, cfg.get("query_present", False), workers, root)
    p = fit_params(cfg)
    temporal = pair_features(table, p).score(p)
    norm = normalize(table, cfg.get("norm", "minmax"))
    gate = load_gate(cfg, table, split)
    scores = combine(table, norm, temporal, cfg.get("a", 1.0), cfg.get("b", 0.0), gate)
    return to_run(table, scores), table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.temporal.rescore")
    ap.add_argument("--run", required=True, type=Path, help="input TREC run (any system)")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="output run (default runs/<split>/<stem>+<system_name>.trec)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-eval", action="store_true", help="only write the run")
    ap.add_argument("--no-log", action="store_true", help="do not append to results/experiments.csv")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    name = f"{args.run.stem}+{cfg['system_name']}"
    out = args.out or REPO_ROOT / "runs" / args.split / f"{name}.trec"
    run, _ = rescore_run(read_run(args.run), args.split, cfg, args.workers)
    write_run(run, out, tag=name.replace("+", "_"), depth=cfg.get("depth", 100))
    config_path = str(args.config)
    desc = f"{cfg.get('description', '')} [input: {args.run.name}]"
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"description": desc, "config": config_path, "input_run": str(args.run)}, indent=2) + "\n")
    print(f"wrote {out}")
    if not args.no_eval:
        result = evaluate_run(read_run(out), args.split)
        print(format_table(result))
        if not args.no_log:
            append_result(result, out.stem, desc, config_path)
            print("appended to results/experiments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

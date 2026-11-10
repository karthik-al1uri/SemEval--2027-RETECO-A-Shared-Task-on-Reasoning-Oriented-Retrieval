"""End-to-end system from one config: first stage -> cross-encoder -> interpolation -> temporal
-> near-duplicate demotion.

    python -m reteco.pipeline --config configs/final.yaml               train + dev runs
    python -m reteco.pipeline --config configs/final.yaml --ablation    + leave-one-out ablation on dev

Stages (every stage's output is recomputed on each call):

  1. sparse     tuned BM25 (sparse config: per-domain k1/b, query cleaning), depth 1000
  2. dense      bi-encoder + exact FAISS search (dense config), depth 1000
  3. fusion     weighted RRF of 1 and 2 (k, w_dense), depth 1000
  4. rerank     cross-encoder on the top `depth` of 3
  5. interp     a * norm(fusion score) + (1 - a) * norm(cross-encoder logit)
  6. temporal   temporal-fit rescorer (reteco.temporal) on the top 100 of 5
  7. dedup      near-duplicate demotion (reteco.analysis_tools.dedup) of the top 100 of 6

Only expensive, input-keyed caches are reused, and each is rebuilt when
missing: sparse term-count indexes (cache/sparse/tf/), document embeddings
(cache/dense/emb/), cross-encoder pair scores (cache/rerank/scores/) and
document time extractions (cache/temporal/doc_times/). With an empty cache/
the same command rebuilds them (several hours: about 2.4 h to embed the corpus
and 1-2 h of cross-encoder scoring on an M4).

Writes runs/<split>/<system_name>.trec (+ .meta.json), scores each split and
appends it to results/experiments.csv. --ablation turns each component off in
turn, keeping every other setting fixed, scores the result on dev and writes
results/ablation.csv (and logs each variant to results/experiments.csv).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .data import DOMAINS, REPO_ROOT, SPLITS, load_queries
from .evaluate import append_result, evaluate_run, format_table
from .runs import _ordered, write_run

ABLATIONS = {
    # name: (what is removed, config edit)
    "no_sparse": ("sparse retrieval (first stage = dense only)", lambda c: c["components"].update(sparse=False)),
    "no_dense": ("dense retrieval (first stage = sparse only)", lambda c: c["components"].update(dense=False)),
    "no_rerank": ("cross-encoder (and so interpolation): fusion -> temporal",
                  lambda c: c["components"].update(rerank=False)),
    "no_interp": ("score interpolation (a = 0: cross-encoder score only)",
                  lambda c: c["components"].update(interpolate=False)),
    "no_temporal": ("temporal rescorer", lambda c: c["components"].update(temporal=False)),
    "no_query_clean": ("query cleaning for the cross-encoder (raw HTML query)",
                       lambda c: c["rerank"].update(query_clean=[])),
    "no_dedup": ("near-duplicate demotion", lambda c: c["components"].update(dedup=False)),
}


class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    def add(self, stage: str, seconds: float):
        self.stages[stage] = self.stages.get(stage, 0.0) + seconds


def load_yaml(path) -> dict:
    return yaml.safe_load(open(REPO_ROOT / path))


# ------------------------------------------------------------- stages --
def sparse_stage(cfg: dict, domains, splits, depth: int, timer: Timer, workers: int = 8):
    from .sparse.index import TFIndex
    from .sparse.retrieve import search, tokenize_queries
    from .sparse.run import domain_params

    scfg = load_yaml(cfg["sparse"]["config"])
    if scfg.get("rm3"):
        raise ValueError("RM3 sparse configs are not supported in the pipeline")
    out = {s: {} for s in splits}
    for domain in domains:
        index = TFIndex.get(domain, scfg.get("corpus", "raw"), workers)
        k1, b = domain_params(scfg, domain)
        weights = index.bm25(k1, b)
        for split in splits:
            qs = load_queries(domain, split)
            t0 = time.perf_counter()
            ranked = search(index, tokenize_queries(qs, scfg.get("query_clean")), k1, b, depth,
                            None, weights=weights)
            timer.add(f"sparse_{split}", time.perf_counter() - t0)
            for q, r in zip(qs, ranked):
                out[split][q.id] = dict(r)
        del index, weights
    print(f"sparse done ({len(domains)} domains)", flush=True)
    return out


def dense_stage(cfg: dict, domains, splits, depth: int, timer: Timer):
    from .dense.encode import encode_domain, encode_queries, is_encoded, load_domain, load_model
    from .dense.retrieve import search_first, search_maxp

    dcfg = load_yaml(cfg["dense"]["config"])
    model = load_model(dcfg["model"], dcfg.get("dtype", "float16"))
    out = {s: {} for s in splits}
    for domain in domains:
        if not is_encoded(dcfg, domain):
            print(f"encoding {domain} (not cached) ...", flush=True)
            encode_domain(dcfg, domain, model)
        emb, doc_ids, doc_index, _ = load_domain(dcfg, domain)
        for split in splits:
            qs = load_queries(domain, split)
            t0 = time.perf_counter()
            q = encode_queries(model, [x.query for x in qs], dcfg)
            if dcfg["doc_mode"] == "maxp":
                res = search_maxp(emb, doc_ids, doc_index, q, depth)
            else:
                res = search_first(emb, doc_ids, q, depth)
            timer.add(f"dense_{split}", time.perf_counter() - t0)
            for x, r in zip(qs, res):
                out[split][x.id] = r
        del emb
    del model
    print(f"dense done ({len(domains)} domains)", flush=True)
    return out


def fusion_stage(cfg: dict, sparse: dict | None, dense: dict | None, depth: int):
    """qid -> [(docid, first-stage score)] in run order."""
    from .dense.fusion import ranks, rrf

    if sparse is not None and dense is not None:
        f = cfg["fusion"]
        fused = rrf([ranks(sparse), ranks(dense)], [1 - f["w_dense"], f["w_dense"]], f["k"], depth)
    else:
        fused = sparse if sparse is not None else dense
    return {q: _ordered(docs, depth) for q, docs in fused.items()}


def rerank_stage(cfg: dict, cands: dict, split: str, timer: Timer, encoder=None):
    from .rerank.cross_encoder import score_candidates

    rcfg = dict(cfg["rerank"], seed=cfg.get("seed", 13))
    scores, secs, encoder = score_candidates(cands, split, rcfg, rcfg["depth"], encoder)
    timer.add(f"rerank_{split}", float(sum(secs.values())))
    return scores, encoder


def temporal_stage(cfg: dict, run: dict, split: str, timer: Timer, workers: int = 8):
    from .temporal.doctimes import get_doc_times, run_doc_ids_by_domain
    from .temporal.rescore import rescore_run

    tcfg = load_yaml(cfg["temporal"]["config"])
    # Document times are a document-side cache (like an index): fill it before timing.
    for dom, ids in sorted(run_doc_ids_by_domain(run).items()):
        get_doc_times(dom, ids, workers)
    t0 = time.perf_counter()
    out, _ = rescore_run(run, split, tcfg, workers)
    timer.add(f"temporal_{split}", time.perf_counter() - t0)
    return out


def dedup_stage(cfg: dict, run: dict, split: str, timer: Timer, depth: int = 100):
    from .analysis_tools.dedup import apply, profile
    from .analysis_tools.errors import fetch_docs

    dcfg = load_yaml(cfg["dedup"]["config"])
    # Rank by the scores as they are written to disk (rounded), so the result is identical
    # to running the rescorer on the written run file.
    run = {q: dict(_ordered(docs, depth)) for q, docs in run.items()}
    # Document texts are document-side data (like an index): read them before timing.
    texts = fetch_docs({d for docs in run.values() for d in docs})
    t0 = time.perf_counter()
    out, _ = apply(profile(run, texts=texts), dcfg["threshold"], dcfg.get("skip", 1),
                   dcfg.get("depth", depth))
    timer.add(f"dedup_{split}", time.perf_counter() - t0)
    return out


# ----------------------------------------------------------- pipeline --
def run_pipeline(cfg: dict, splits, domains, workers: int = 8, shared: dict | None = None):
    """Returns {split: run} and the stage timer. `shared` caches first-stage outputs between
    ablation variants (they do not depend on the later stages)."""
    from .rerank.interpolate import combine

    comp = cfg["components"]
    depth1 = cfg["fusion"].get("depth", 1000)
    timer = Timer()
    shared = shared if shared is not None else {}
    first = {}
    for stage, fn in (("sparse", sparse_stage), ("dense", dense_stage)):
        if not comp.get(stage, True):
            first[stage] = None
            continue
        key = (stage, tuple(domains))
        runs_, t = shared.get(key, ({}, Timer()))
        todo = [s for s in splits if s not in runs_]
        if todo:
            args = (cfg, domains, todo, depth1, t) + ((workers,) if stage == "sparse" else ())
            runs_.update(fn(*args))
            shared[key] = (runs_, t)
        first[stage] = runs_
        for s in splits:
            timer.add(f"{stage}_{s}", t.stages.get(f"{stage}_{s}", 0.0))
    sparse, dense = first["sparse"], first["dense"]

    runs, encoder = {}, shared.get("encoder")
    for split in splits:
        t0 = time.perf_counter()
        cands = fusion_stage(cfg, sparse and sparse[split], dense and dense[split], depth1)
        timer.add(f"fusion_{split}", time.perf_counter() - t0)
        if comp.get("rerank", True):
            scores, encoder = rerank_stage(cfg, cands, split, timer, encoder)
            shared["encoder"] = encoder
            t0 = time.perf_counter()
            if comp.get("interpolate", True):
                ic = cfg["interpolate"]
                run = combine(cands, scores, cfg["rerank"]["depth"], ic["a"], ic["norm"])
            else:
                run = combine(cands, scores, cfg["rerank"]["depth"], 0.0)
            timer.add(f"interp_{split}", time.perf_counter() - t0)
        else:
            run = {q: dict(docs[:100]) for q, docs in cands.items()}
        if comp.get("temporal", True):
            run = temporal_stage(cfg, run, split, timer, workers)
        if comp.get("dedup", True) and cfg.get("dedup"):
            run = dedup_stage(cfg, run, split, timer)
        runs[split] = run
    return runs, timer


def latency(timer: Timer, split: str, n_queries: int) -> dict[str, float]:
    per = {k.rsplit("_", 1)[0]: 1000 * v / n_queries for k, v in timer.stages.items()
           if k.endswith(f"_{split}")}
    per["total"] = sum(per.values())
    return {k: round(v, 1) for k, v in per.items()}


def describe(cfg: dict) -> str:
    c = cfg["components"]
    parts = []
    first = [n for n in ("sparse", "dense") if c.get(n, True)]
    parts.append("+".join(first) + (f" RRF k={cfg['fusion']['k']} w_dense={cfg['fusion']['w_dense']}"
                                    if len(first) == 2 else ""))
    if c.get("rerank", True):
        r = cfg["rerank"]
        qc = "clean query" if r.get("query_clean") else "raw query"
        parts.append(f"{r['model'].split('/')[-1]} top {r['depth']} ({qc})")
        if c.get("interpolate", True):
            parts.append(f"interp a={cfg['interpolate']['a']} {cfg['interpolate']['norm']}")
    if c.get("temporal", True):
        parts.append(f"temporal {Path(cfg['temporal']['config']).stem}")
    if c.get("dedup", True) and cfg.get("dedup"):
        parts.append(f"dedup {Path(cfg['dedup']['config']).stem}")
    return " -> ".join(parts)


def write_and_score(cfg: dict, runs: dict, timer: Timer, name: str, config_path: str, domains,
                    log: bool = True) -> dict:
    results = {}
    for split, run in runs.items():
        path = write_run(run, REPO_ROOT / "runs" / split / f"{name}.trec", tag=name, depth=100)
        lat = latency(timer, split, len(run))
        desc = describe(cfg)
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"description": desc, "config": config_path, "domains": list(domains),
             "latency_ms_per_query": lat}, indent=2) + "\n")
        res = evaluate_run(run, split, list(domains))
        print(f"\n{name} [{split}]  latency ms/query {lat}\n{format_table(res)}", flush=True)
        if log:
            append_result(res, name, desc, config_path, latency_ms=lat["total"])
        results[split] = (res, lat)
    return results


def ablation(cfg: dict, config_path: str, domains, workers: int, log: bool,
             shared: dict | None = None) -> Path:
    shared = shared if shared is not None else {}
    rows = []
    variants = [("full", "nothing (full system)", cfg)]
    for name, (what, edit) in ABLATIONS.items():
        c = copy.deepcopy(cfg)
        edit(c)
        variants.append((name, what, c))
    full = None
    for name, what, c in variants:
        print(f"\n=== ablation {name}: remove {what}", flush=True)
        runs, timer = run_pipeline(c, ["dev"], domains, workers, shared)
        sys_name = cfg["system_name"] + ("" if name == "full" else f"_{name}")
        res, lat = write_and_score(c, runs, timer, sys_name, config_path, domains,
                                   log=log and name != "full")["dev"]
        if full is None:
            full = res
        rows.append({"variant": name, "removed": what, "split": "dev",
                     "macro_ndcg@10": f"{res['macro_ndcg@10']:.4f}",
                     "delta_vs_full": f"{res['macro_ndcg@10'] - full['macro_ndcg@10']:+.4f}",
                     "macro_ndcg@10_no_iota": f"{np.mean([r['ndcg@10'] for d, r in res['domains'].items() if d != 'iota']):.4f}",
                     "macro_recall@100": f"{res['macro_recall@100']:.4f}",
                     **{d: f"{res['domains'][d]['ndcg@10']:.4f}" for d in domains},
                     "latency_ms_per_query": lat["total"], "system_name": sys_name})
    path = REPO_ROOT / "results" / "ablation.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.pipeline")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=None, choices=DOMAINS)
    ap.add_argument("--ablation", action="store_true", help="also run the dev ablation")
    ap.add_argument("--ablation-only", action="store_true", help="only run the dev ablation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    cfg.setdefault("components", {})
    domains = args.domains or DOMAINS
    shared: dict = {}
    if not args.ablation_only:
        runs, timer = run_pipeline(cfg, [s for s in SPLITS if s in args.splits], domains, args.workers, shared)
        write_and_score(cfg, runs, timer, cfg["system_name"], args.config, domains, not args.no_log)
    if args.ablation or args.ablation_only:
        ablation(cfg, args.config, domains, args.workers, not args.no_log, shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

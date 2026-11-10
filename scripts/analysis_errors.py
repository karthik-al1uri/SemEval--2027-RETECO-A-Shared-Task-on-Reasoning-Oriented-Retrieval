"""Select 25 dev queries where hybrid_best fails and collect evidence for each.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_errors.py \
        --dossier /tmp/error_dossier.md

Pool: dev queries with hybrid_best nDCG@10 < 0.05. Sample: 25 queries, at least
one per domain in the pool, the rest proportional to the domain's share of the
pool (seed 13). Needs results/analysis_per_query.csv.

Writes
    results/analysis_error_candidates.csv  per sampled query: automatic signals
    a Markdown dossier (query, gold and top-ranked documents) for manual labeling

The manual categories live in results/analysis_error_labels.csv (hand-written,
one row per sampled query); this script never overwrites that file.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from reteco.analysis_tools.errors import (content_tokens, doc_prox, fetch_docs, ranks,
                                          stratified_sample, years)
from reteco.data import DOMAINS, REPO_ROOT, load_qrels, load_queries
from reteco.runs import read_run
from reteco.temporal.extract import clean_text

OUT = REPO_ROOT / "results"
K = 25
THRESH = 0.05
CAND = REPO_ROOT / "cache" / "dense" / "candidates"   # depth-1000 lists, read only
BOILER = re.compile(r"cookie|subscribe|newsletter|all rights reserved|copyright|privacy policy|"
                    r"click here|sign up|log ?in|terms of (use|service)|retrieved \d|archived from|"
                    r"isbn|press enter|full text not available|permission|license", re.I)


def snippet(text: str, n: int = 60) -> str:
    words = clean_text(text).split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", type=Path, required=True,
                    help="where to write the Markdown dossier (keep it out of git: it quotes dev gold text)")
    args = ap.parse_args()

    with open(OUT / "analysis_per_query.csv", newline="") as f:
        pq = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    info = {r["qid"]: r for r in pq}
    pool: dict[str, list[str]] = {d: [] for d in DOMAINS}
    for r in pq:
        if float(r["hybrid_best|ndcg@10"]) < THRESH:
            pool[r["domain"]].append(r["qid"])
    print("pool:", sum(map(len, pool.values())), {d: len(v) for d, v in pool.items() if v})
    sample = stratified_sample(pool, K)

    queries = {q.id: q.query for d in DOMAINS for q in load_queries(d, "dev")}
    qrels = {}
    for d in DOMAINS:
        qrels.update(load_qrels(d, "dev"))
    runs = {s: ranks(read_run(REPO_ROOT / "runs" / "dev" / f"{s}.trec"))
            for s in ("hybrid_best", "sparse_best", "dense_best")}
    deep = {}
    for s in ("sparse_best", "dense_best"):
        p = CAND / s / "dev.trec"
        if p.exists():
            deep[s] = ranks(read_run(p))

    need = set()
    for q in sample:
        need |= set(qrels[q])
        need |= {d for d, r in runs["hybrid_best"][q].items() if r <= 3}
    docs = fetch_docs(need)

    rows, md = [], ["# Error dossier (dev, hybrid_best nDCG@10 < 0.05)\n"]
    for q in sample:
        r = info[q]
        text = queries[q]
        q_iv = years(text, present=True)
        q_tok = content_tokens(text)
        gold = sorted(qrels[q])
        top = sorted((rk, d) for d, rk in runs["hybrid_best"][q].items() if rk <= 10)

        def rank_of(s, d, table=runs):
            return table.get(s, {}).get(q, {}).get(d)

        g_rows = []
        for g in gold:
            gt = docs.get(g, "")
            g_tok = content_tokens(gt)
            g_rows.append({
                "id": g, "words": len(gt.split()),
                "q_cover": len(q_tok & g_tok) / max(1, len(q_tok)),
                "boiler": len(BOILER.findall(gt)),
                "prox": doc_prox(q_iv, years(gt)),
                "r_h": rank_of("hybrid_best", g), "r_s": rank_of("sparse_best", g),
                "r_d": rank_of("dense_best", g),
                "r_s1000": rank_of("sparse_best", g, deep), "r_d1000": rank_of("dense_best", g, deep),
            })
        top_prox = [doc_prox(q_iv, years(docs.get(d, ""))) for rk, d in top[:3]]
        top_prox = [x for x in top_prox if x is not None]
        gold_prox = [g["prox"] for g in g_rows if g["prox"] is not None]
        best_h = min((g["r_h"] for g in g_rows if g["r_h"]), default=None)
        best_deep = min((x for g in g_rows for x in (g["r_s1000"], g["r_d1000"]) if x),
                        default=None)
        row = {
            "domain": r["domain"], "qid": q, "type": r["type"], "anchored": r["anchored"],
            "n_words": r["n_words"], "n_gold": len(gold),
            "best_gold_rank_hybrid100": best_h or "", "best_gold_rank_depth1000": best_deep or "",
            "gold_mean_words": f"{np.mean([g['words'] for g in g_rows]):.0f}",
            "gold_mean_query_cover": f"{np.mean([g['q_cover'] for g in g_rows]):.3f}",
            "gold_boiler_hits": sum(g["boiler"] for g in g_rows),
            "gold_time_prox": f"{np.mean(gold_prox):.3f}" if gold_prox else "",
            "top3_time_prox": f"{np.mean(top_prox):.3f}" if top_prox else "",
            "sparse_ndcg@10": r["sparse_best|ndcg@10"], "dense_ndcg@10": r["dense_best|ndcg@10"],
        }
        rows.append(row)

        md.append(f"\n## {q} ({r['domain']}, type {r['type']}, {r['n_words']} words)\n")
        md.append(f"query intervals: {q_iv[:8]}\n\n> {snippet(text, 220)}\n")
        md.append(f"signals: {row}\n")
        md.append("\n**gold**\n")
        for g in g_rows:
            md.append(f"- `{g['id']}` words={g['words']} cover={g['q_cover']:.2f} "
                      f"boiler={g['boiler']} prox={g['prox']} ranks h/s/d={g['r_h']}/{g['r_s']}/"
                      f"{g['r_d']} deep s/d={g['r_s1000']}/{g['r_d1000']}\n"
                      f"  > {snippet(docs.get(g['id'], ''), 90)}\n")
        md.append("\n**hybrid top 3**\n")
        for rk, d in top[:3]:
            md.append(f"- {rk}. `{d}` intervals={years(docs.get(d, ''))[:6]}\n"
                      f"  > {snippet(docs.get(d, ''), 60)}\n")

    with open(OUT / "analysis_error_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    args.dossier.write_text("".join(md), encoding="utf-8")
    print("wrote", OUT / "analysis_error_candidates.csv", "and", args.dossier)


if __name__ == "__main__":
    main()

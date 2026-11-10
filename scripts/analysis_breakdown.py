"""Per-query, per-type and per-domain nDCG@10 / Recall@100 of the existing runs.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_breakdown.py

Needs results/analysis_query_types.csv (scripts/analysis_query_types.py).

Writes
    results/analysis_per_query.csv    split, domain, qid, type, facets, every system's scores
    results/analysis_per_domain.csv   dev, per domain and macro, with query counts
    results/analysis_per_type.csv     per type (and explicit-anchor facet), query means
    results/analysis_type_deltas.csv  per-type paired differences between systems, 95% CI

Per-type scores are plain means over the queries of that type (micro), because
many (type, domain) cells hold 0-3 queries; per-domain scores are the usual
domain means and the macro row is the official macro average.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from reteco.analysis_tools import SYSTEMS
from reteco.analysis_tools.perquery import per_query_systems
from reteco.analysis_tools.qtypes import TYPES
from reteco.analysis_tools.significance import query_bootstrap_ci
from reteco.data import DOMAINS, REPO_ROOT

OUT = REPO_ROOT / "results"
METRICS = ["ndcg@10", "recall@100"]

# (system, baseline) pairs whose per-type differences are reported.
DELTA_PAIRS = [
    ("bm25+temporal_fit_present", "bm25"),
    ("sparse_best", "bm25"),
    ("dense_best", "sparse_best"),
    ("hybrid_best", "sparse_best"),
    ("hybrid_best", "dense_best"),
]


def col(system: str, metric: str) -> str:
    return f"{system}|{metric}"


def load_types() -> dict[tuple[str, str], dict]:
    with open(OUT / "analysis_query_types.csv", newline="") as f:
        return {(r["split"], r["qid"]): r for r in csv.DictReader(f)}


def build_rows(split: str, types) -> list[dict]:
    per = per_query_systems(SYSTEMS, split)
    rows = []
    for i, base in enumerate(per[SYSTEMS[0]]):
        t = types[(split, base["qid"])]
        row = {"split": split, "domain": base["domain"], "qid": base["qid"],
               "type": t["type"], "cv_pred": t["cv_pred"], "anchored": int(t["anchored"]),
               "n_words": int(t["n_words"])}
        for s in SYSTEMS:
            assert per[s][i]["qid"] == base["qid"]
            for m in METRICS:
                row[col(s, m)] = per[s][i][m]
        rows.append(row)
    return rows


def write_csv(path, header, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    types = load_types()
    rows = {s: build_rows(s, types) for s in ("train", "dev")}

    score_cols = [col(s, m) for s in SYSTEMS for m in METRICS]
    write_csv(OUT / "analysis_per_query.csv",
              ["split", "domain", "qid", "type", "cv_pred", "anchored", "n_words"] + score_cols,
              [[r["split"], r["domain"], r["qid"], r["type"], r["cv_pred"], r["anchored"],
                r["n_words"]] + [fmt(r[c]) for c in score_cols]
               for sp in ("train", "dev") for r in rows[sp]])

    # ---- per domain (dev), plus macro
    dev = rows["dev"]
    by_dom = defaultdict(list)
    for r in dev:
        by_dom[r["domain"]].append(r)
    out = []
    for d in DOMAINS:
        rs = by_dom[d]
        out.append([d, len(rs)] + [fmt(np.mean([r[c] for r in rs])) for c in score_cols])
    out.append(["macro", len(dev)] + [fmt(np.mean([np.mean([r[c] for r in by_dom[d]])
                                                    for d in DOMAINS])) for c in score_cols])
    write_csv(OUT / "analysis_per_domain.csv", ["domain", "n"] + score_cols, out)

    # ---- per type: dev (predicted), train (guidance label), train (CV prediction)
    views = [("dev", "predicted", "type"), ("train", "guidance", "type"),
             ("train", "predicted_oof", "cv_pred")]
    out = []
    for split, source, key in views:
        rs_all = rows[split]
        groups = [(t, [r for r in rs_all if r[key] == t]) for t in TYPES]
        groups += [("anchored", [r for r in rs_all if r["anchored"]]),
                   ("not_anchored", [r for r in rs_all if not r["anchored"]]),
                   ("all", rs_all)]
        for t, rs in groups:
            if not rs:
                continue
            best = max(SYSTEMS, key=lambda s: np.mean([r[col(s, 'ndcg@10')] for r in rs]))
            out.append([split, source, t, len(rs),
                        fmt(np.mean([r["domain"] == "history" for r in rs])),
                        fmt(np.mean([r["anchored"] for r in rs])), best]
                       + [fmt(np.mean([r[c] for r in rs])) for c in score_cols])
    write_csv(OUT / "analysis_per_type.csv",
              ["split", "type_source", "type", "n", "share_history", "share_anchored",
               "best_ndcg_system"] + score_cols, out)

    # ---- per-type paired differences (dev predicted types, train guidance types)
    out = []
    for split, source, key in views[:2]:
        rs_all = rows[split]
        for t in TYPES + ["anchored", "not_anchored", "all"]:
            if t == "anchored":
                rs = [r for r in rs_all if r["anchored"]]
            elif t == "not_anchored":
                rs = [r for r in rs_all if not r["anchored"]]
            elif t == "all":
                rs = rs_all
            else:
                rs = [r for r in rs_all if r[key] == t]
            for a, b in DELTA_PAIRS:
                for m in METRICS:
                    d = np.array([r[col(a, m)] - r[col(b, m)] for r in rs])
                    lo, hi = query_bootstrap_ci(d)
                    out.append([split, source, t, len(rs), a, b, m, fmt(d.mean()), fmt(lo), fmt(hi),
                                int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())])
    write_csv(OUT / "analysis_type_deltas.csv",
              ["split", "type_source", "type", "n", "system", "baseline", "metric", "delta",
               "ci_low", "ci_high", "n_better", "n_worse", "n_tied"], out)


if __name__ == "__main__":
    main()

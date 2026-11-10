"""TRAIN-only analysis: how often queries / gold docs carry time expressions,
and whether gold documents fit the query's time better than the non-gold
documents in the input run's top 100.

Gold labels (qrels_train) are used here only to describe the data; nothing
from this script feeds a prediction.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo \\
    python -m reteco.temporal.analyze --run runs/train/bm25.trec
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data import DOMAINS, load_qrels
from ..runs import read_run
from .doctimes import CACHE_DIR, get_doc_times
from .fit import min_dists
from .rescore import build_table


def _arr(iv, drop=("present",)):
    return np.asarray([(s, e) for s, e, k in iv if k not in drop], dtype=float).reshape(-1, 2)


def _pair_stats(qa: np.ndarray, dt: list, tau: float) -> tuple[float, float, float] | None:
    da = _arr(dt)
    if not len(qa) or not len(da):
        return None
    dm, qm = min_dists(qa, da)
    best = dm.min()
    sim = 0.5 * (np.exp(-dm / tau).mean() + np.exp(-qm / tau).mean())
    return float(best == 0), float(best <= tau), float(sim)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def analyze(run_path: Path, split: str = "train", tau: float = 10.0, workers: int = 8) -> dict:
    assert split == "train", "analysis uses gold labels: train only"
    table = build_table(read_run(run_path), split, 100, False, workers)
    out = {}
    kinds_q, kinds_gold = Counter(), Counter()
    for dom in DOMAINS:
        qrels = load_qrels(dom, split)
        idx = [i for i, d in enumerate(table.domains) if d == dom]
        gold_ids = {d for q in qrels.values() for d, r in q.items() if r > 0}
        gold_times = get_doc_times(dom, gold_ids, workers, verbose=False)
        s = Counter()
        g_rows, n_rows = [], []  # (overlap, near, sim) for gold-any / non-gold top-100
        aucs, aucs_all = [], []
        for i in idx:
            q = table.qids[i]
            qa = _arr(table.q_times[i])
            s["queries"] += 1
            s["q_time"] += bool(len(qa))
            kinds_q.update(k for _, _, k in table.q_times[i])
            gold = {d for d, r in qrels.get(q, {}).items() if r > 0}
            for d in gold:
                s["gold"] += 1
                s["gold_time"] += bool(len(_arr(gold_times[d])))
                kinds_gold.update(k for _, _, k in gold_times[d])
            a, b = table.offsets[i], table.offsets[i + 1]
            nongold = [j for j in range(a, b) if table.docids[j] not in gold]
            s["cand"] += len(nongold)
            s["cand_time"] += sum(bool(len(_arr(table.d_times[j]))) for j in nongold)
            if not len(qa):
                continue
            pos_top = [table.docids[j] for j in range(a, b) if table.docids[j] in gold]
            sims_gold_all, sims_gold_top, sims_neg = [], [], []
            for d in gold:
                st = _pair_stats(qa, gold_times[d], tau)
                if st:
                    g_rows.append(st)
                sims_gold_all.append(st[2] if st else 0.0)
                if d in pos_top:
                    sims_gold_top.append(st[2] if st else 0.0)
            for j in nongold:
                st = _pair_stats(qa, table.d_times[j], tau)
                if st:
                    n_rows.append(st)
                sims_neg.append(st[2] if st else 0.0)
            auc = _auc(sims_gold_top, sims_neg)
            if auc is not None:
                aucs.append(auc)
            auc = _auc(sims_gold_all, sims_neg)
            if auc is not None:
                aucs_all.append(auc)
        g, n = np.asarray(g_rows).reshape(-1, 3), np.asarray(n_rows).reshape(-1, 3)
        out[dom] = {
            "queries": s["queries"],
            "q_time": s["q_time"] / s["queries"],
            "gold_time": s["gold_time"] / max(s["gold"], 1),
            "cand_time": s["cand_time"] / max(s["cand"], 1),
            "gold_pairs": len(g), "cand_pairs": len(n),
            "gold_overlap": float(g[:, 0].mean()) if len(g) else None,
            "cand_overlap": float(n[:, 0].mean()) if len(n) else None,
            "gold_near": float(g[:, 1].mean()) if len(g) else None,
            "cand_near": float(n[:, 1].mean()) if len(n) else None,
            "gold_sim": float(g[:, 2].mean()) if len(g) else None,
            "cand_sim": float(n[:, 2].mean()) if len(n) else None,
            "auc_top100": float(np.mean(aucs)) if aucs else None, "auc_queries": len(aucs),
            "auc_allgold": float(np.mean(aucs_all)) if aucs_all else None,
        }
    out["_kinds"] = {"query": dict(kinds_q), "gold": dict(kinds_gold)}
    return out


def to_markdown(res: dict) -> str:
    f = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    g = lambda x: "–" if x is None else f"{x:.3f}"
    lines = ["| Domain | Queries | Q w/ time | Gold w/ time | Non-gold top-100 w/ time "
             "| Overlap gold / non-gold | Within τ gold / non-gold | Mean sim gold / non-gold | AUC (top-100) | AUC queries |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d in DOMAINS:
        r = res[d]
        lines.append(
            f"| {d} | {r['queries']} | {f(r['q_time'])} | {f(r['gold_time'])} | {f(r['cand_time'])} "
            f"| {f(r['gold_overlap'])} / {f(r['cand_overlap'])} | {f(r['gold_near'])} / {f(r['cand_near'])} "
            f"| {g(r['gold_sim'])} / {g(r['cand_sim'])} | {g(r['auc_top100'])} | {r['auc_queries']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.temporal.analyze")
    ap.add_argument("--run", type=Path, default=Path("runs/train/bm25.trec"))
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    res = analyze(args.run, "train", args.tau, args.workers)
    path = CACHE_DIR / f"analysis_train_{args.run.stem}.json"
    path.write_text(json.dumps(res, indent=2))
    print(to_markdown(res))
    print("\nexpression kinds:", json.dumps(res["_kinds"]))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate near-duplicate demotion on a base run: gains, significance, gold demotions.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_dedup.py
    # the same on the final system (runs/<split>/final+dedup.trec, configs/final_dedup_end.yaml)
    python scripts/analysis_dedup.py --base final --variant dedup:final_dedup_end --tag _final

Needs the runs written by `python -m reteco.analysis_tools.dedup` (by default with
configs/dedup.yaml and configs/dedup_end.yaml on hybrid_best). Qrels are read only to score runs and to audit which
demoted documents were gold (train is the requested check; dev is descriptive).

Writes (with --tag T, results/analysis_dedup{T}.csv etc.)
    results/analysis_dedup.csv          per variant and split: macro with/without iota,
                                        deltas, CI, p, queries better/worse, demotions
    results/analysis_dedup_domains.csv  per-domain nDCG@10 deltas
    results/analysis_dedup_gold.csv     every demoted gold document, with its partner
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import yaml

from reteco.analysis_tools.dedup import is_dup, profile, ranked
from reteco.analysis_tools.perquery import macro, per_query
from reteco.analysis_tools.significance import bootstrap_ci, macro_diff, randomization_test
from reteco.data import DOMAINS, REPO_ROOT, load_qrels
from reteco.runs import read_run

OUT = REPO_ROOT / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hybrid_best", help="base run name in runs/<split>/")
    ap.add_argument("--variant", action="append", metavar="RUN_SUFFIX:CONFIG",
                    help="runs/<split>/<base>+<RUN_SUFFIX>.trec made with configs/<CONFIG>.yaml "
                         "(default: dedup:dedup and dedup_end:dedup_end)")
    ap.add_argument("--tag", default="", help="suffix for the output CSV names")
    args = ap.parse_args(argv)
    BASE = args.base
    variants = [tuple(v.split(":", 1)) for v in (args.variant or ["dedup:dedup", "dedup_end:dedup_end"])]
    summary, domains, gold_rows = [], [], []
    for split in ("train", "dev"):
        base_run = read_run(REPO_ROOT / "runs" / split / f"{BASE}.trec")
        base_pq = per_query(base_run, split)
        prof = profile(base_run)
        qrels = {}
        for d in DOMAINS:
            qrels.update(load_qrels(d, split))
        for v, cfg_name in variants:
            cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{cfg_name}.yaml").read_text())
            new_run = read_run(REPO_ROOT / "runs" / split / f"{BASE}+{v}.trec")
            pq = per_query(new_run, split)
            assert [r["qid"] for r in pq] == [r["qid"] for r in base_pq]
            a = np.array([r["ndcg@10"] for r in base_pq])
            b = np.array([r["ndcg@10"] for r in pq])
            doms = [r["domain"] for r in pq]
            keep = np.array([d != "iota" for d in doms])
            dk = [d for d in doms if d != "iota"]
            lo, hi = bootstrap_ci(b, a, doms)
            n_dem = n_dem10 = 0
            for q, p in prof.items():
                dup = is_dup(p, cfg["threshold"])
                new_order = ranked(new_run[q])
                pos = {d: i for i, d in enumerate(new_order)}
                for i in np.flatnonzero(dup):
                    n_dem += 1
                    n_dem10 += int(i < 10)
                    doc = p["docs"][i]
                    if qrels.get(q, {}).get(doc, 0) <= 0:
                        continue
                    j = p["strict_of"][i] if p["strict"][i] else p["argprev"][i]
                    partner = p["docs"][j]
                    gold_rows.append({
                        "split": split, "variant": v, "domain": doc.split("/")[0],
                        "qid": q, "doc": doc, "orig_rank": i + 1, "new_rank": pos[doc] + 1,
                        "partner": partner, "partner_rank": j + 1,
                        "partner_is_gold": int(qrels[q].get(partner, 0) > 0),
                        "similarity": "strict" if p["strict"][i] else f"{p['maxprev'][i]:.3f}",
                    })
            g = [r for r in gold_rows if r["split"] == split and r["variant"] == v]
            summary.append({
                "split": split, "variant": v, "threshold": cfg["threshold"], "skip": cfg["skip"],
                "base_macro": f"{macro(base_pq, 'ndcg@10'):.4f}",
                "macro": f"{macro(pq, 'ndcg@10'):.4f}",
                "delta": f"{macro_diff(b, a, doms):+.4f}", "ci_low": f"{lo:+.4f}",
                "ci_high": f"{hi:+.4f}", "p_randomization": f"{randomization_test(b, a, doms):.4f}",
                "base_macro_no_iota": f"{macro([r for r in base_pq if r['domain'] != 'iota'], 'ndcg@10'):.4f}",
                "macro_no_iota": f"{macro([r for r in pq if r['domain'] != 'iota'], 'ndcg@10'):.4f}",
                "delta_no_iota": f"{macro_diff(b[keep], a[keep], dk):+.4f}",
                "p_no_iota": f"{randomization_test(b[keep], a[keep], dk):.4f}",
                "micro_delta": f"{float(np.mean(b - a)):+.4f}",
                "n_better": int((b > a).sum()), "n_worse": int((b < a).sum()),
                "n_same": int((b == a).sum()),
                "recall@100_delta": f"{macro(pq, 'recall@100') - macro(base_pq, 'recall@100'):+.4f}",
                "demoted": n_dem, "demoted_from_top10": n_dem10,
                "demoted_gold": len(g), "demoted_gold_from_top10": sum(r["orig_rank"] <= 10 for r in g),
                "demoted_gold_partner_gold": sum(r["partner_is_gold"] for r in g),
            })
            print(summary[-1])
            for d in DOMAINS:
                m = np.array([x == d for x in doms])
                domains.append({"split": split, "variant": v, "domain": d, "n": int(m.sum()),
                                "base": f"{a[m].mean():.4f}", "new": f"{b[m].mean():.4f}",
                                "delta": f"{(b[m] - a[m]).mean():+.4f}",
                                "n_better": int((b[m] > a[m]).sum()),
                                "n_worse": int((b[m] < a[m]).sum())})

    t = args.tag
    for name, rows in ((f"analysis_dedup{t}.csv", summary), (f"analysis_dedup{t}_domains.csv", domains),
                       (f"analysis_dedup{t}_gold.csv", gold_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()

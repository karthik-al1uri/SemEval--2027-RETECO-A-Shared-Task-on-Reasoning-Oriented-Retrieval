"""R2-E4: first-stage comparison of LLM-rewrite variants on the tune part (no cross-encoder).

    python -m reteco.playground.e4_rewrite_eval configs/pg2_e3.yaml configs/pg2_e4q.yaml configs/pg2_e4h.yaml

For each config: fusion (with its rewrite lists, if any) on all train queries, scored on tune
(968 queries): macro nDCG@10 and Recall@100 of the fused list, plus Recall@200 (the cross-
encoder's depth). Also reports the LLM generation time per query. Holdout is not scored here.
"""
from __future__ import annotations

import sys

import numpy as np

from ..data import DOMAINS
from ..pipeline import Timer, fusion_stage, load_yaml, rewrite_stage
from .common import _qrels, part_qids, score_set
from .harness import preload_shared


def recall_at(run: dict, keep: set[str], k: int) -> float:
    per = []
    for d in DOMAINS:
        vals = []
        for q, rel in _qrels(d).items():
            if q not in keep:
                continue
            gold = {x for x, r in rel.items() if r > 0}
            if gold:
                top = [x for x, _ in run.get(q, [])[:k]]
                vals.append(len(gold & set(top)) / len(gold))
        per.append(np.mean(vals) if vals else 0.0)
    return float(np.mean(per))


def main(argv=None) -> int:
    tune = part_qids("tune")
    for path in (argv or sys.argv[1:]):
        cfg = load_yaml(path)
        cfg.setdefault("components", {})
        shared = preload_shared(cfg)
        sp = shared[("sparse", tuple(DOMAINS))][0]["train"]
        dn = shared[("dense", tuple(DOMAINS))][0]["train"]
        extra, llm_ms = None, 0.0
        if cfg.get("rewrite"):
            t = Timer()
            extra = rewrite_stage(cfg, DOMAINS, ["train"], 1000, t)["train"]
            llm_ms = 1000 * t.stages.get("llm_train", 0.0) / len(sp)
        cands = fusion_stage(cfg, sp, dn, 1000, extra)
        top100 = {q: dict(v[:100]) for q, v in cands.items()}
        r = score_set(top100, tune, "tune")
        print(f"{path}: tune nDCG@10 {r['macro']:.4f} (no iota {r['no_iota']:.4f})  R@100 {r['recall@100']:.4f}  "
              f"R@200 {recall_at(cands, tune, 200):.4f}  llm {llm_ms:.0f} ms/query", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How much of each system's dev top 10 is taken by near-duplicate documents?

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_neardup.py

v1.1 removed exact duplicates (whitespace-insensitive). Texts that differ only in
punctuation or case remain separate documents. A top-10 slot counts as a
near-duplicate when its document matches a HIGHER-ranked document of the same
list under either test:
    strict   identical after lowercasing and dropping everything but [a-z0-9]
    shingle  Jaccard similarity of word 5-gram sets >= 0.8
No qrels are used. Writes results/analysis_neardup.csv (per system and domain).
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict

from reteco.analysis_tools import STAGES, SYSTEMS
from reteco.analysis_tools.errors import fetch_docs, ranks
from reteco.data import DOMAINS, REPO_ROOT, load_queries
from reteco.runs import read_run

OUT = REPO_ROOT / "results"
TOP = 10
JACCARD = 0.8
_NON = re.compile(r"[^a-z0-9]+")


def strict_key(t: str) -> str:
    return _NON.sub("", t.lower())


def shingles(t: str, n: int = 5) -> set:
    w = _NON.sub(" ", t.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def main() -> None:
    qdom = {q.id: d for d in DOMAINS for q in load_queries(d, "dev")}
    tops = {}
    need = set()
    systems = SYSTEMS[:-1] + STAGES[-1:] + SYSTEMS[-1:]   # final before and after demotion
    for s in systems:
        rk = ranks(read_run(REPO_ROOT / "runs" / "dev" / f"{s}.trec"))
        tops[s] = {q: [d for d, r in sorted(v.items(), key=lambda x: x[1]) if r <= TOP]
                   for q, v in rk.items()}
        for lst in tops[s].values():
            need |= set(lst)
    print("fetching", len(need), "documents")
    docs = fetch_docs(need)
    keys = {d: strict_key(t) for d, t in docs.items()}
    sh = {d: shingles(t) for d, t in docs.items()}

    out = []
    for s in systems:
        agg = defaultdict(lambda: [0, 0, 0, 0, 0])   # queries, slots, strict, shingle, q_with_dup
        for q, lst in tops[s].items():
            dom = qdom[q]
            n_strict = n_sh = 0
            for i, d in enumerate(lst):
                prev = lst[:i]
                if any(keys[d] == keys[p] for p in prev):
                    n_strict += 1
                if any(len(sh[d] & sh[p]) / max(1, len(sh[d] | sh[p])) >= JACCARD for p in prev):
                    n_sh += 1
            for key in (dom, "all"):
                a = agg[key]
                a[0] += 1
                a[1] += len(lst)
                a[2] += n_strict
                a[3] += n_sh
                a[4] += int(n_sh > 0)
        for dom in DOMAINS + ["all"]:
            qn, slots, st, shn, qd = agg[dom]
            out.append([s, dom, qn, slots, st, shn, f"{st / slots:.4f}", f"{shn / slots:.4f}",
                        f"{qd / qn:.4f}"])
            if dom == "all":
                print(f"{s:<28} strict {st / slots:.3f}  shingle {shn / slots:.3f}  "
                      f"queries with >=1 near-dup {qd / qn:.3f}")
    with open(OUT / "analysis_neardup.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "domain", "queries", "top10_slots", "strict_dups", "shingle_dups",
                    "strict_share", "shingle_share", "queries_with_dup_share"])
        w.writerows(out)
    print("wrote", OUT / "analysis_neardup.csv")


if __name__ == "__main__":
    main()

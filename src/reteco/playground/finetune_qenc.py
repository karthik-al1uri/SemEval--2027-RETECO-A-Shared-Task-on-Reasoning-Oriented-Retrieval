"""E8: fine-tune only the query side of the bge-small bi-encoder against frozen document
embeddings (no corpus re-encoding).

    python -m reteco.playground.finetune_qenc --name bge_q --epochs 4

Data: tune_train queries (as in E1), positives from qrels_train, negatives = the same
near-duplicate-filtered fusion top 200 as E1 (reteco.playground.finetune_ce.build_examples).
Document vectors are the cached bge-small first-512 embeddings and never change, so all
negatives of a query fit in one step: loss = softmax cross-entropy of q.d / tau over
[positive, all negatives]. After every epoch the query encoder is saved and scored on tune_val:
dense-only nDCG@10 and hybrid nDCG@10 (RRF with the cached sparse run, the champion's k and
w_dense). The pipeline uses it via `query_model` in the dense config.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np

from ..data import DOMAINS, REPO_ROOT, load_queries
from ..dense.encode import load_domain
from ..dense.fusion import ranks, rrf
from ..dense.retrieve import search_first
from ..dense.text import clean_query
from ..pipeline import load_yaml
from .common import score_set, subsplit_tune
from .finetune_ce import build_examples
from .harness import preload_shared

OUT = REPO_ROOT / "cache" / "playground" / "qenc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.finetune_qenc")
    ap.add_argument("--config", default="configs/pg_e5.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from sentence_transformers import SentenceTransformer

    random.seed(13)
    torch.manual_seed(13)
    cfg = load_yaml(args.config)
    dcfg = load_yaml(cfg["dense"]["config"])
    tune_train, tune_val = subsplit_tune()
    ex, _ = build_examples(cfg, tune_train, 200, 0.6)

    # frozen doc vectors for every example doc
    need: dict[str, set] = {}
    for e in ex:
        for d in e["pos"] + e["neg"]:
            need.setdefault(d.split("/", 1)[0], set()).add(d)
    vec = {}
    for dom in DOMAINS:
        emb, ids, _, _ = load_domain(dcfg, dom)
        row = {d: i for i, d in enumerate(ids)}
        for d in need.get(dom, ()):
            vec[d] = emb[row[d]].copy()
        del emb
    print(f"{len(vec)} doc vectors", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(dcfg["model"], device=dev)
    model.max_seq_length = dcfg.get("query_max_tokens", 512)
    prefix = dcfg.get("query_prefix", "")
    groups = []
    for e in ex:
        q = prefix + clean_query(e["query"])
        negs = torch.tensor(np.stack([vec[d] for d in e["neg"]]), dtype=torch.float32)
        for p in e["pos"]:
            groups.append((q, torch.tensor(vec[p], dtype=torch.float32), negs))

    # tune_val evaluation helpers
    shared = preload_shared(cfg)
    sparse = shared[("sparse", tuple(DOMAINS))][0]["train"]
    val_q = {d: [q for q in load_queries(d, "train") if q.id in tune_val] for d in DOMAINS}

    def evaluate(m) -> tuple[float, float]:
        m.eval()
        dense = {}
        with torch.no_grad():
            for dom, qs in val_q.items():
                qe = m.encode([prefix + clean_query(q.query) for q in qs], batch_size=32,
                              normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                emb, ids, _, _ = load_domain(dcfg, dom)
                for q, r in zip(qs, search_first(emb, ids, qe, 1000)):
                    dense[q.id] = r
                del emb
        sp = {q: sparse[q] for q in dense}
        f = cfg["fusion"]
        hyb = rrf([ranks(sp), ranks(dense)], [1 - f["w_dense"], f["w_dense"]], f["k"], 100)
        m.train()
        top100 = {q: dict(sorted(v.items(), key=lambda x: -x[1])[:100]) for q, v in dense.items()}
        return score_set(top100, tune_val, "v")["macro"], score_set(hyb, tune_val, "v")["macro"]

    d0, h0 = evaluate(model)
    print(f"epoch 0: dense {d0:.4f} hybrid {h0:.4f}", flush=True)
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = [{"epoch": 0, "dense": d0, "hybrid": h0}]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    rng = random.Random(13)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(groups)
        losses = []
        for i in range(0, len(groups), args.batch):
            chunk = groups[i:i + args.batch]
            feats = model.tokenize([g[0] for g in chunk])
            feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in feats.items()}
            qe = torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)
            loss = 0.0
            for j, (_, pos, negs) in enumerate(chunk):
                docs = torch.cat([pos[None], negs]).to(dev)
                logits = (docs @ qe[j]) / args.tau
                loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long, device=dev))
            loss = loss / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if len(losses) % 25 == 0:
                print(f"  epoch {epoch} step {len(losses)}: loss {np.mean(losses[-25:]):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        d, h = evaluate(model)
        ck = out_dir / f"epoch{epoch}"
        model.save(str(ck))
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "dense": d, "hybrid": h})
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} dense {d:.4f} hybrid {h:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    (out_dir / "log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

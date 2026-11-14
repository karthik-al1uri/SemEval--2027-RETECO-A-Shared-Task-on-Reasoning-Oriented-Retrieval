"""Round 2, R2-E5: fine-tune the full bge-small bi-encoder (query AND document side).

    python -m reteco.playground.finetune_bienc --name bge_full --config configs/pg2_e3.yaml

Data (TRAIN only): tune_train queries (reteco.playground.common.subsplit_tune), positives from
qrels_train, hard negatives from the current champion's fusion top 200 minus near-copies of a
positive (strict text match or word 5-gram Jaccard >= 0.6; reteco.playground.finetune_ce.
build_examples). Unlike E8 (query tower only, frozen cached document vectors), both towers
are trained (bge-small shares one encoder; queries get the instruction prefix, documents not).

Loss: InfoNCE over the step's documents: for each query, its positive against its own
`n_neg` hard negatives plus every other document of the step (in-batch negatives), tau 0.02.
Hyperparameters are fixed in advance (no tuning on tune_val): lr 2e-5 with 10% warmup and
linear decay, 4 queries x (1 + 7) documents per step, documents cut to 256 tokens in training
(inference encodes the first 512), 1 epoch, seed 13.

Validation after training (--validate):
  pool   for every tune_val query, re-rank the champion's fusion top 200 by the new model vs
         the champion's dense model (fine-tuned query encoder + cached document vectors);
         dense-only nDCG@10 within the pool, 194 queries, all domains
  small  full retrieval on the 5 small domains (encodes their corpora, ~162k docs, under the
         model's own embedding cache key): dense-only and hybrid nDCG@10 on their tune_val
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time

import numpy as np

from ..data import DOMAINS, REPO_ROOT, SMALL_DOMAINS, load_queries
from ..dense.text import clean_query
from ..pipeline import load_yaml
from .common import score_set, subsplit_tune
from .finetune_ce import build_examples
from .harness import preload_shared, train_candidates

OUT = REPO_ROOT / "cache" / "playground2" / "bienc"


def train(args) -> str:
    import torch
    from sentence_transformers import SentenceTransformer

    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    cfg = load_yaml(args.config)
    dcfg = load_yaml(cfg["dense"]["config"])
    tune_train, _ = subsplit_tune()
    ex, docs = build_examples(cfg, tune_train, 200, 0.6)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(dcfg["model"], device=dev)
    prefix = dcfg.get("query_prefix", "")
    groups = [(prefix + clean_query(e["query"]), p, e["neg"]) for e in ex for p in e["pos"]]
    steps = math.ceil(len(groups) / args.queries_per_step) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(13)
    model.train()

    def embed(texts, max_len):
        model.max_seq_length = max_len
        f = model.tokenize(texts)
        f = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in f.items()}
        return torch.nn.functional.normalize(model(f)["sentence_embedding"], dim=-1)

    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(groups)
        losses = []
        for i in range(0, len(groups), args.queries_per_step):
            chunk = groups[i:i + args.queries_per_step]
            dl = []
            for _, p, negs in chunk:
                ns = rng.sample(negs, min(args.n_neg, len(negs)))
                dl += [p] + ns
            qe = embed([q for q, _, _ in chunk], dcfg.get("query_max_tokens", 512))
            de = embed([docs[d] for d in dl], args.doc_tokens)
            logits = qe @ de.T / args.tau
            # target: each query's own positive; a positive of another query that is also
            # gold for this query is not used as a negative
            starts, pos_idx = [], 0
            for _, _, negs in chunk:
                starts.append(pos_idx)
                pos_idx += 1 + min(args.n_neg, len(negs))
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for a, (_, pa, _) in enumerate(chunk):
                for b, (_, pb, _) in enumerate(chunk):
                    if a != b and pb == pa:
                        mask[a, starts[b]] = True
            logits = logits.masked_fill(mask, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(starts, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{steps} loss {np.mean(losses[-50:]):.4f} "
                      f"({el / step:.2f}s/step, eta {(steps - step) * el / step / 60:.0f} min)", flush=True)
        ck = out_dir / f"epoch{epoch}"
        model.save(str(ck))
        print(f"saved {ck} (epoch mean loss {np.mean(losses):.4f})", flush=True)
    (out_dir / "meta.json").write_text(json.dumps({**vars(args), "groups": len(groups),
                                                   "steps": steps}, indent=2) + "\n")
    return str((out_dir / f"epoch{args.epochs}").relative_to(REPO_ROOT))


def validate(args, model_path: str) -> None:
    import torch
    from ..dense.encode import encode, encode_domain, encode_queries, load_domain, load_model
    from ..dense.fusion import ranks, rrf
    from ..dense.retrieve import search_first
    from ..analysis_tools.errors import fetch_docs

    cfg = load_yaml(args.config)
    dcfg = load_yaml(cfg["dense"]["config"])
    _, tune_val = subsplit_tune()
    base_doc = load_model(dcfg["model"], dcfg.get("dtype", "float16"))
    base_q = load_model(str(REPO_ROOT / dcfg["query_model"]), dcfg.get("dtype", "float16")) \
        if dcfg.get("query_model") else base_doc
    new = load_model(str(REPO_ROOT / model_path), dcfg.get("dtype", "float16"))
    ndcfg = {k: v for k, v in dcfg.items() if k != "query_model"}
    ndcfg["model"] = model_path

    # pool: re-rank the champion's fusion top 200 of every tune_val query
    cands = {q: v for q, v in train_candidates(cfg, 200).items() if q in tune_val}
    texts = {q.id: q.query for d in DOMAINS for q in load_queries(d, "train") if q.id in tune_val}
    dtext = fetch_docs({d for v in cands.values() for d, _ in v})
    old_run, new_run = {}, {}
    for dom in DOMAINS:
        qs = sorted(q for q in cands if q.startswith(dom) or cands[q][0][0].startswith(dom + "/"))
        if not qs:
            continue
        emb, ids, _, _ = load_domain(dcfg, dom)
        row = {d: i for i, d in enumerate(ids)}
        qo = encode_queries(base_q, [texts[q] for q in qs], dcfg)
        qn = encode_queries(new, [texts[q] for q in qs], ndcfg)
        for k, q in enumerate(qs):
            dl = [d for d, _ in cands[q]]
            old_run[q] = dict(zip(dl, (emb[[row[d] for d in dl]] @ qo[k]).tolist()))
        new.max_seq_length = dcfg.get("doc_max_tokens", 512)
        alld = sorted({d for q in qs for d, _ in cands[q]})
        de = dict(zip(alld, encode(new, [dtext[d] for d in alld], dcfg["batch_size"])))
        for k, q in enumerate(qs):
            dl = [d for d, _ in cands[q]]
            new_run[q] = dict(zip(dl, (np.stack([de[d] for d in dl]) @ qn[k]).tolist()))
        del emb
        print(f"pool {dom}: {len(qs)} queries, {len(alld)} docs", flush=True)
    so, sn = score_set(old_run, tune_val, "pool_old"), score_set(new_run, tune_val, "pool_new")
    print(f"POOL tune_val dense-in-top200 nDCG@10: champion {so['macro']:.4f}  new {sn['macro']:.4f}  "
          f"(no iota {so['no_iota']:.4f} -> {sn['no_iota']:.4f})", flush=True)
    if args.pool_only:
        return

    # small domains: full retrieval with re-encoded corpora
    shared = preload_shared(cfg)
    sparse = shared[("sparse", tuple(DOMAINS))][0]["train"]
    f = cfg["fusion"]
    res = {}
    for name, qm, dc in (("champion", base_q, dcfg), ("new", new, ndcfg)):
        dense = {}
        for dom in SMALL_DOMAINS:
            encode_domain(dc, dom, new if name == "new" else base_doc)
            emb, ids, _, _ = load_domain(dc, dom)
            qs = [q for q in load_queries(dom, "train") if q.id in tune_val]
            for q, r in zip(qs, search_first(emb, ids, encode_queries(qm, [x.query for x in qs], dc), 1000)):
                dense[q.id] = r
            del emb
        hyb = rrf([ranks({q: sparse[q] for q in dense}), ranks(dense)], [1 - f["w_dense"], f["w_dense"]],
                  f["k"], 100)
        top = {q: dict(sorted(v.items(), key=lambda x: -x[1])[:100]) for q, v in dense.items()}
        res[name] = (score_set(top, tune_val, name, SMALL_DOMAINS), score_set(hyb, tune_val, name, SMALL_DOMAINS))
        print(f"SMALL {name}: dense {res[name][0]['macro']:.4f} hybrid {res[name][1]['macro']:.4f}  "
              + " ".join(f"{d[:4]}={v:.3f}" for d, v in res[name][1]["domains"].items()), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.finetune_bienc")
    ap.add_argument("--config", default="configs/pg2_e3.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--n-neg", type=int, default=7)
    ap.add_argument("--queries-per-step", type=int, default=4)
    ap.add_argument("--doc-tokens", type=int, default=256)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--model", default=None, help="skip training, validate this model")
    args = ap.parse_args(argv)
    path = args.model or train(args)
    if args.validate or args.model:
        validate(args, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

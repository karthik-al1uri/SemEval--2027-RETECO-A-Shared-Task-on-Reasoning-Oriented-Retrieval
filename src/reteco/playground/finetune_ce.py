"""Fine-tune the MiniLM cross-encoder on tune-split queries with hard negatives (E1).

    python -m reteco.playground.finetune_ce --name minilm_ft --epochs 2

Data (TRAIN only; never holdout, never dev):
  queries    tune_train = 80% of the tune part per domain (reteco.playground.common.subsplit_tune);
             the other 20% (tune_val) selects the epoch and the interpolation weight
  positives  qrels_train documents with rel > 0 for those queries
  negatives  the fusion (hybrid) top 200 of the query minus the positives and minus near-
             duplicates of a positive (strict text match or word 5-gram Jaccard >= 0.6), so
             unjudged copies of a gold document are not trained as negatives

Loss: listwise softmax cross-entropy over groups of 1 positive + `n_neg` negatives
(negatives resampled every epoch), the usual recipe for cross-encoder rerankers.
Inputs are built exactly like inference (reteco.rerank.cross_encoder): cleaned query cut
to 256 tokens, document cut to 4000 characters, pair truncated to 512 tokens.
Checkpoints: cache/playground/ce/<name>/epoch<k>/ (loadable with `rerank.model`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict

import numpy as np

from ..analysis_tools.dedup import jaccard_matrix, shingle_hashes, strict_key
from ..analysis_tools.errors import fetch_docs
from ..data import DOMAINS, REPO_ROOT, load_qrels, load_queries
from ..pipeline import load_yaml
from .common import subsplit_tune
from .harness import train_candidates

OUT = REPO_ROOT / "cache" / "playground" / "ce"


def build_examples(cfg: dict, qids: set[str], depth: int, dup_jaccard: float, domains=None):
    cands = train_candidates(cfg, depth)
    qrels = {}
    for d in (domains or DOMAINS):
        qrels.update(load_qrels(d, "train"))
    texts = {}
    for d in (domains or DOMAINS):
        for q in load_queries(d, "train"):
            if q.id in qids:
                texts[q.id] = q.query
    use = sorted(q for q in texts if q in cands)
    need = set()
    for q in use:
        need |= {d for d, _ in cands[q]}
        need |= {d for d, r in qrels.get(q, {}).items() if r > 0}
    t0 = time.time()
    docs = fetch_docs(need)
    print(f"loaded {len(docs)} docs in {time.time() - t0:.0f}s", flush=True)
    ex, n_dup = [], 0
    for q in use:
        pos = [d for d, r in qrels.get(q, {}).items() if r > 0 and d in docs]
        if not pos:
            continue
        cand = [d for d, _ in cands[q] if d not in set(pos)]
        vocab: dict[str, int] = {}
        lst = pos + cand
        sh = [shingle_hashes(docs[d], vocab) for d in lst]
        J = jaccard_matrix(sh)[len(pos):, :len(pos)]
        pkeys = {strict_key(docs[d]) for d in pos}
        neg = []
        for i, d in enumerate(cand):
            if J[i].max() >= dup_jaccard or strict_key(docs[d]) in pkeys:
                n_dup += 1
                continue
            neg.append(d)
        ex.append({"qid": q, "query": texts[q], "pos": pos, "neg": neg})
    print(f"{len(ex)} queries, {sum(len(e['pos']) for e in ex)} positives, "
          f"{n_dup} near-duplicate negatives dropped", flush=True)
    return ex, docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.finetune_ce")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-neg", type=int, default=15)
    ap.add_argument("--groups-per-step", type=int, default=2)
    ap.add_argument("--neg-depth", type=int, default=200)
    ap.add_argument("--dup-jaccard", type=float, default=0.6)
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after this many steps")
    ap.add_argument("--base-model", default=None, help="model to start from (default: rerank.model)")
    ap.add_argument("--max-query-tokens", type=int, default=None, help="override rerank.max_query_tokens")
    ap.add_argument("--doc-chars", type=int, default=None, help="override rerank.doc_chars")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = load_yaml(args.config)
    rcfg = dict(cfg["rerank"])
    if args.base_model:
        rcfg["model"] = args.base_model
    if args.max_query_tokens:
        rcfg["max_query_tokens"] = args.max_query_tokens
    if args.doc_chars:
        rcfg["doc_chars"] = args.doc_chars
    tune_train, tune_val = subsplit_tune()
    ex, docs = build_examples(cfg, tune_train, args.neg_depth, args.dup_jaccard, args.domains)

    from ..rerank.cross_encoder import device
    from ..sparse.clean import clean_query
    dev = device()
    tok = AutoTokenizer.from_pretrained(rcfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(rcfg["model"], dtype=torch.float32).to(dev)
    model.train()

    def prep_query(text):
        q = clean_query(text, rcfg.get("query_clean") or [])
        ids = tok(q, add_special_tokens=False, truncation=True,
                  max_length=rcfg.get("max_query_tokens", 256))["input_ids"]
        return tok.decode(ids)

    for e in ex:
        e["q"] = prep_query(e["query"])
    dchars = rcfg.get("doc_chars", 4000)

    groups_per_epoch = sum(len(e["pos"]) for e in ex)
    steps_per_epoch = math.ceil(groups_per_epoch / args.groups_per_step)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(args).items()}
    meta.update(base_model=rcfg["model"], n_queries=len(ex), groups_per_epoch=groups_per_epoch,
                steps_per_epoch=steps_per_epoch, tune_train=len(tune_train), tune_val=len(tune_val))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    step, t0 = 0, time.time()
    rng = random.Random(seed)
    for epoch in range(1, args.epochs + 1):
        groups = []
        for e in ex:
            for p in e["pos"]:
                negs = rng.sample(e["neg"], min(args.n_neg, len(e["neg"])))
                if len(negs) < args.n_neg:  # rare: pad by repetition
                    negs += rng.choices(e["neg"], k=args.n_neg - len(negs))
                groups.append((e["q"], [p] + negs))
        rng.shuffle(groups)
        losses = []
        for i in range(0, len(groups), args.groups_per_step):
            chunk = groups[i:i + args.groups_per_step]
            qs = [q for q, ds in chunk for _ in ds]
            ds = [docs[d][:dchars] for _, dl in chunk for d in dl]
            enc = tok(qs, ds, padding=True, truncation="only_second",
                      max_length=rcfg.get("max_length", 512), return_tensors="pt").to(dev)
            logits = model(**enc).logits[:, 0].view(len(chunk), -1)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(chunk), dtype=torch.long,
                                                                        device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total} loss {np.mean(losses[-100:]):.4f} "
                      f"({el / step:.2f}s/step, eta {(total - step) * el / step / 60:.0f} min)", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        ck = out_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        print(f"saved {ck} (epoch mean loss {np.mean(losses):.4f})", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

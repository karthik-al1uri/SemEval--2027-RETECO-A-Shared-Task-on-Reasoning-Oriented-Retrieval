"""Round 2, R2-E4: rewrite each query with a small local instruction model (no API).

    python -m reteco.playground.llm_rewrite --mode query --splits train dev
    python -m reteco.playground.llm_rewrite --mode hyde --splits train --limit 32   (timing)

Two modes, from the query text only (the system input at prediction time):
  query  a short focused search query (<= 16 words) for the question in the post
  hyde   a short hypothetical reference passage (~80 words) that would answer it (HyDE)

The post is cleaned like sparse_best (code blocks, HTML, URLs, sign-offs dropped) and cut to
`--max-input-tokens`. Greedy decoding, fp16 on MPS, batched with left padding.
Outputs are cached per (model, mode, split, domain) under cache/llm_local/ as jsonl
{"id", "text", "seconds"} (seconds = the batch's generation wall time spread evenly over
its queries, counted as latency by the pipeline); a rerun only generates the missing queries.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..data import DOMAINS, REPO_ROOT, load_queries
from ..sparse.clean import clean_query

CACHE = REPO_ROOT / "cache" / "llm_local"

PROMPTS = {
    "query": ("Below is a question someone posted on a Q&A forum. Write one short search query "
              "(at most 16 words) that a search engine could use to find documents answering it. "
              "Keep names, places, dates and technical terms. Output only the query.\n\nPost:\n{post}"),
    "hyde": ("Below is a question someone posted on a Q&A forum. Write a short factual passage "
             "(about 80 words) that answers it, in the style of an encyclopedia or reference document. "
             "Mention the specific names, terms and dates involved. Output only the passage.\n\nPost:\n{post}"),
}
MAX_NEW = {"query": 32, "hyde": 140}


def cache_path(model: str, mode: str, split: str, domain: str) -> Path:
    return CACHE / model.replace("/", "-") / mode / split / f"{domain}.jsonl"


def load_rewrites(model: str, mode: str, split: str, domains=None, seconds: bool = False):
    """qid -> rewrite text (or qid -> (text, generation seconds) with seconds=True)."""
    out = {}
    for d in domains or DOMAINS:
        p = cache_path(model, mode, split, d)
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = (r["text"], r.get("seconds", 0.0)) if seconds else r["text"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.playground.llm_rewrite")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", required=True, choices=list(PROMPTS))
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-input-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="timing: stop after this many new queries")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(13)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()

    def prompt(text: str) -> str:
        post = clean_query(text)
        ids = tok(post, add_special_tokens=False, truncation=True,
                  max_length=args.max_input_tokens)["input_ids"]
        msgs = [{"role": "user", "content": PROMPTS[args.mode].format(post=tok.decode(ids))}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    n_new, t_start, n_tok = 0, time.time(), 0
    for split in args.splits:
        for d in args.domains or DOMAINS:
            p = cache_path(args.model, args.mode, split, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            done = set(load_rewrites(args.model, args.mode, split, [d]))
            todo = [q for q in load_queries(d, split) if q.id not in done]
            # similar lengths together: less padding
            todo.sort(key=lambda q: len(q.query))
            with open(p, "a", encoding="utf-8") as f:
                for i in range(0, len(todo), args.batch):
                    chunk = todo[i:i + args.batch]
                    t0 = time.perf_counter()
                    enc = tok([prompt(q.query) for q in chunk], return_tensors="pt", padding=True).to(dev)
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=MAX_NEW[args.mode], do_sample=False,
                                             pad_token_id=tok.pad_token_id)
                    gen = out[:, enc["input_ids"].shape[1]:].cpu()
                    share = (time.perf_counter() - t0) / len(chunk)  # batch wall time, spread evenly
                    n_tok += int((gen != tok.pad_token_id).sum())
                    for q, g in zip(chunk, gen):
                        text = tok.decode(g, skip_special_tokens=True).strip().strip('"').strip()
                        f.write(json.dumps({"id": q.id, "text": text, "seconds": round(share, 4)}) + "\n")
                    f.flush()
                    n_new += len(chunk)
                    el = time.time() - t_start
                    print(f"{split} {d}: {n_new} new in {el:.0f}s ({el / n_new:.2f} s/query, "
                          f"{n_tok / el:.0f} tok/s)", flush=True)
                    if args.limit and n_new >= args.limit:
                        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

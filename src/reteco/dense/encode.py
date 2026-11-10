"""Bi-encoder wrapper and resumable per-domain document encoding.

Document embeddings are cached under cache/dense/<key>/<domain>/:

    emb.npy        float16 [n_units, dim], L2-normalised (units = docs or passages)
    doc_ids.json   document ids, in corpus order
    doc_index.npy  int32 [n_units] unit -> row in doc_ids (passage mode only)
    meta.json      sizes and encoding cost

While a domain is being encoded, finished shards of `shard_size` documents are
written to <domain>/shards/ with atomic renames. Re-running skips finished
shards, so an interruption loses at most one shard. Once every shard is done
they are merged into the files above and the shard folder is removed.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..data import data_dir, iter_corpus
from . import CACHE_DIR
from .text import clean_query, split_passages


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def hardware() -> dict:
    import torch
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "device": device(),
            "mps_available": torch.backends.mps.is_available()}
    if platform.system() == "Darwin":
        def sysctl(k):
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        info["chip"] = sysctl("machdep.cpu.brand_string")
        info["ram_gb"] = int(sysctl("hw.memsize")) / 2**30
        info["cpu_cores"] = int(sysctl("hw.ncpu"))
    return info


def load_model(name: str, dtype: str = "float16", max_seq_length: int = 512):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(13)
    model = SentenceTransformer(name, device=device(),
                                model_kwargs={"torch_dtype": getattr(torch, dtype)})
    model.max_seq_length = max_seq_length
    return model


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def encode_queries(model, texts: list[str], cfg: dict) -> np.ndarray:
    old = model.max_seq_length
    model.max_seq_length = cfg.get("query_max_tokens", 512)
    try:
        prefix = cfg.get("query_prefix", "")
        return encode(model, [prefix + clean_query(t) for t in texts], cfg["batch_size"])
    finally:
        model.max_seq_length = old


# ------------------------------------------------------------------ cache --
def cache_key(cfg: dict) -> str:
    """Folder name for a document-embedding config (everything that changes the vectors)."""
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", cfg["model"]).strip("-")
    if cfg["doc_mode"] == "first":
        mode = f"first{cfg['doc_max_tokens']}"
    elif cfg["doc_mode"] == "maxp":
        mode = f"maxp{cfg['passage_tokens']}s{cfg['passage_stride']}"
    else:
        raise ValueError(f"unknown doc_mode {cfg['doc_mode']!r}")
    key = f"{slug}__{mode}__{cfg.get('dtype', 'float16')}"
    if cfg.get("doc_prefix"):
        key += "__p-" + re.sub(r"[^A-Za-z0-9]+", "", cfg["doc_prefix"])[:20]
    return key


def domain_dir(cfg: dict, domain: str) -> Path:
    return CACHE_DIR / "emb" / cache_key(cfg) / domain


def is_encoded(cfg: dict, domain: str) -> bool:
    return (domain_dir(cfg, domain) / "meta.json").exists()


def _doc_chunks(domain: str, size: int):
    ids, texts = [], []
    for doc_id, text in iter_corpus(domain):
        ids.append(doc_id)
        texts.append(text)
        if len(ids) == size:
            yield ids, texts
            ids, texts = [], []
    if ids:
        yield ids, texts


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def encode_domain(cfg: dict, domain: str, model=None, log=None) -> Path:
    """Encode one domain's corpus into the cache (resumable). Returns the domain folder."""
    log = log or (lambda msg: print(msg, flush=True))
    out = domain_dir(cfg, domain)
    if (out / "meta.json").exists():
        return out
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    maxp = cfg["doc_mode"] == "maxp"
    unit_len = cfg["passage_tokens"] if maxp else cfg["doc_max_tokens"]
    t_start = time.time()
    n_shards = 0
    for k, (ids, texts) in enumerate(_doc_chunks(domain, cfg["shard_size"])):
        n_shards += 1
        path = shard_dir / f"{k:05d}.npz"
        if path.exists():
            continue
        if model is None:
            model = load_model(cfg["model"], cfg.get("dtype", "float16"), unit_len)
        model.max_seq_length = unit_len
        prefix = cfg.get("doc_prefix", "")
        t0 = time.time()
        if maxp:
            # Leave room for [CLS]/[SEP] so a window is never truncated again.
            units, owner = split_passages(texts, model.tokenizer, cfg["passage_tokens"] - 2,
                                          cfg["passage_stride"])
        else:
            units, owner = texts, list(range(len(texts)))
        emb = encode(model, [prefix + u for u in units], cfg["batch_size"])
        secs = time.time() - t0
        _atomic_save_npz(path, emb=emb.astype(np.float16), ids=np.array(ids),
                         owner=np.array(owner, dtype=np.int32), seconds=np.array(secs))
        log(f"  {domain} shard {k:4d}: {len(ids)} docs, {len(units)} units, "
            f"{len(ids) / secs:6.1f} docs/s")
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # Merge shards.
    embs, doc_ids, doc_index, seconds = [], [], [], 0.0
    for k in range(n_shards):
        with np.load(shard_dir / f"{k:05d}.npz") as z:
            embs.append(z["emb"])
            doc_index.append(z["owner"] + len(doc_ids))
            doc_ids.extend(z["ids"].tolist())
            seconds += float(z["seconds"])
    emb = np.concatenate(embs)
    np.save(out / "emb.npy", emb)
    if maxp:
        np.save(out / "doc_index.npy", np.concatenate(doc_index).astype(np.int32))
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))
    meta = {"model": cfg["model"], "doc_mode": cfg["doc_mode"], "unit_tokens": unit_len,
            "dtype": cfg.get("dtype", "float16"), "n_docs": len(doc_ids), "n_units": int(emb.shape[0]),
            "dim": int(emb.shape[1]), "encode_seconds": round(seconds, 1),
            "docs_per_second": round(len(doc_ids) / seconds, 1) if seconds else None,
            "data_dir": str(data_dir()), "wall_seconds_last_run": round(time.time() - t_start, 1),
            "hardware": hardware()}
    if maxp:
        meta.update(passage_tokens=cfg["passage_tokens"], passage_stride=cfg["passage_stride"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.rmtree(shard_dir)
    log(f"{domain}: {meta['n_docs']} docs, {meta['n_units']} units, "
        f"{meta['encode_seconds']}s encode ({meta['docs_per_second']} docs/s)")
    return out


def load_domain(cfg: dict, domain: str):
    """(emb float32 [n_units, dim], doc_ids, doc_index or None, meta)."""
    d = domain_dir(cfg, domain)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{domain} not encoded for {cache_key(cfg)}; run encoding first")
    emb = np.load(d / "emb.npy").astype(np.float32)
    doc_ids = json.loads((d / "doc_ids.json").read_text())
    doc_index = np.load(d / "doc_index.npy") if (d / "doc_index.npy").exists() else None
    return emb, doc_ids, doc_index, json.loads((d / "meta.json").read_text())

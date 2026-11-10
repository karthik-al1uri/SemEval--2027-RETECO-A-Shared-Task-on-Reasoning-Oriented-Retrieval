# Dense retrieval and hybrid fusion (Sub-track 1a)

Dense first-stage retrieval with off-the-shelf bi-encoders (no fine-tuning), exact FAISS search per domain, and weighted reciprocal rank fusion (RRF) with sparse retrieval. All numbers are macro nDCG@10 (official metric) and macro Recall@100 over the 13 domains unless a table says "5 small domains". Everything was chosen on **train**; dev is reported, never used for selection.

**Headline (dev, 13 domains):** baseline BM25 0.1148 nDCG@10 / 0.4045 R@100 → `dense_best` 0.2765 / 0.6259 → `hybrid_best` (dense + tuned sparse) **0.3189 / 0.7158**. The hybrid's depth-1000 list holds 92% of dev gold documents (BM25: 66%).

## 1. Hardware and encoding cost

| | |
| --- | --- |
| Machine | Apple M4 (10 CPU cores), 16 GB unified memory |
| Software | Python 3.13.11, torch 2.15.0, sentence-transformers 6.2.0, faiss-cpu 1.15.1 |
| Accelerator | torch MPS available and used for all encoding |

`python -m reteco.dense.bench` measures throughput on 2,000 documents drawn from all 13 domains in proportion to domain size (so the length mix matches the corpus: median 207 tokens, mean 552, 14% of documents longer than 510 tokens). Batch size 64; batch 128 was slower at 512 tokens.

| Model | Precision | Max tokens | Docs/s | Est. full corpus (1.17M docs) |
| --- | --- | ---: | ---: | ---: |
| bge-small-en-v1.5 (33M) | fp16 | 512 | 146 | 2.2 h |
| bge-small-en-v1.5 | fp16 | 256 | 227 | 1.4 h |
| bge-small-en-v1.5 | fp32 | 512 | 119 | 2.7 h |
| bge-base-en-v1.5 (110M) | fp16 | 512 | 47 | 6.9 h |
| bge-base-en-v1.5 | fp16 | 256 | 71 | 4.6 h |
| bge-base-en-v1.5 | fp32 | 512 | 39 | 8.4 h |
| e5-base-v2 (110M) | fp16 | 512 | 48 | 6.8 h |
| e5-base-v2 | fp16 | 256 | 71 | 4.6 h |
| e5-base-v2 | fp32 | 512 | 39 | 8.4 h |

fp16 and fp32 embeddings agree to mean cosine 1.000 for all three models, so fp16 is used throughout.

**Measured** encoding (from `cache/dense/emb/*/<domain>/meta.json`):

| Config | Domain | Docs | Units | Encode (s) | Docs/s |
| --- | --- | ---: | ---: | ---: | ---: |
| BAAI-bge-base-en-v1.5__first512__float16 | cardano | 47,509 | 47,509 | 1004 | 47.3 |
| BAAI-bge-base-en-v1.5__first512__float16 | iota | 9,680 | 9,680 | 223 | 43.5 |
| BAAI-bge-base-en-v1.5__first512__float16 | law | 36,110 | 36,110 | 699 | 51.7 |
| BAAI-bge-base-en-v1.5__first512__float16 | quant | 25,040 | 25,040 | 551 | 45.4 |
| BAAI-bge-base-en-v1.5__first512__float16 | workplace | 43,584 | 43,584 | 812 | 53.7 |
| **BAAI-bge-base-en-v1.5__first512__float16** | **total** | **161,923** | **161,923** | **3288** (0.91 h) | **49.3** |
| BAAI-bge-small-en-v1.5__first512__float16 | bitcoin | 76,706 | 76,706 | 660 | 116.2 |
| BAAI-bge-small-en-v1.5__first512__float16 | cardano | 47,509 | 47,509 | 358 | 132.7 |
| BAAI-bge-small-en-v1.5__first512__float16 | economics | 74,692 | 74,692 | 561 | 133.1 |
| BAAI-bge-small-en-v1.5__first512__float16 | genealogy | 136,101 | 136,101 | 1032 | 131.9 |
| BAAI-bge-small-en-v1.5__first512__float16 | history | 200,252 | 200,252 | 1366 | 146.6 |
| BAAI-bge-small-en-v1.5__first512__float16 | hsm | 169,552 | 169,552 | 1375 | 123.3 |
| BAAI-bge-small-en-v1.5__first512__float16 | iota | 9,680 | 9,680 | 72 | 133.9 |
| BAAI-bge-small-en-v1.5__first512__float16 | law | 36,110 | 36,110 | 269 | 134.3 |
| BAAI-bge-small-en-v1.5__first512__float16 | monero | 53,955 | 53,955 | 424 | 127.2 |
| BAAI-bge-small-en-v1.5__first512__float16 | politics | 135,207 | 135,207 | 877 | 154.1 |
| BAAI-bge-small-en-v1.5__first512__float16 | quant | 25,040 | 25,040 | 200 | 125.2 |
| BAAI-bge-small-en-v1.5__first512__float16 | travel | 158,771 | 158,771 | 1029 | 154.3 |
| BAAI-bge-small-en-v1.5__first512__float16 | workplace | 43,584 | 43,584 | 292 | 149.4 |
| **BAAI-bge-small-en-v1.5__first512__float16** | **total** | **1,167,159** | **1,167,159** | **8516** (2.37 h) | **137.1** |
| BAAI-bge-small-en-v1.5__maxp256s192__float16 | cardano | 47,509 | 119,545 | 549 | 86.5 |
| BAAI-bge-small-en-v1.5__maxp256s192__float16 | iota | 9,680 | 37,843 | 217 | 44.7 |
| BAAI-bge-small-en-v1.5__maxp256s192__float16 | law | 36,110 | 102,783 | 468 | 77.1 |
| BAAI-bge-small-en-v1.5__maxp256s192__float16 | quant | 25,040 | 63,489 | 292 | 85.8 |
| BAAI-bge-small-en-v1.5__maxp256s192__float16 | workplace | 43,584 | 96,114 | 433 | 100.6 |
| **BAAI-bge-small-en-v1.5__maxp256s192__float16** | **total** | **161,923** | **419,774** | **1959** (0.54 h) | **82.6** |

The full bge-small first-512 encode took 2.37 h, within 7% of the 2.2 h estimate. Embeddings are stored as fp16: 0.9 GB for the full bge-small corpus. Exact search is cheap: loading plus flat inner-product search for all 1,730 queries takes about 30 s for all 13 domains.

## 2. Method

- **Documents.** `doc_mode: first`: one vector per document from its first 512 tokens. `doc_mode: maxp`: the document is cut into 256-token windows every 192 tokens (64-token overlap) on the tokenizer's character offsets, each window is encoded, and the document score is the max over its windows.
- **Queries.** Only `id` and `query` are used. Queries are StackExchange posts with an HTML body, so tags are stripped and entities unescaped, then the text is truncated to 512 tokens. bge models get the instruction `Represent this sentence for searching relevant passages: ` (with vs without compared on train, below).
- **Search.** Embeddings are L2-normalised, and a FAISS `IndexFlatIP` per domain gives exact cosine search. MaxP uses an exact segmented max over all passage scores. Each dense system writes a top-100 run to `runs/` and a depth-1000 candidate run to `cache/dense/candidates/<system>/`.
- **Fusion.** `score(d) = w_sparse / (k + rank_sparse(d)) + w_dense / (k + rank_dense(d))` over depth-1000 candidate lists, with `w_sparse = 1 - w_dense`; a missing document contributes 0. (k, w_dense) is tuned on train macro nDCG@10 over k ∈ {1, 2, 5, 10, 20, 30, 60, 100, 200} × w_dense ∈ {0, 0.1, …, 1}.
- **Sparse inputs.** `bm25` is the baseline, with candidates scored from the existing `cache/bm25/` indexes (read-only). `sparse_best` is the sparse agent's tuned BM25 (per-domain k1/b, query cleaning), with depth-1000 candidates produced by running that agent's own code against its cached indexes (`src/reteco/dense/sparse_candidates.py`).

## 3. Results, 13 domains

| System | train nDCG@10 | train R@100 | dev nDCG@10 | dev R@100 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.1076 | 0.3365 | 0.1148 | 0.4045 |
| sparse_best | 0.2435 | 0.5995 | 0.2541 | 0.6531 |
| dense_bge_small_noinst | 0.2069 | 0.5845 | 0.2619 | 0.6076 |
| dense_bge_small | 0.2261 | 0.6152 | 0.2765 | 0.6259 |
| dense_best | 0.2261 | 0.6152 | 0.2765 | 0.6259 |
| hybrid_rrf60 | 0.2302 | 0.6031 | 0.2738 | 0.6469 |
| hybrid_bm25_tuned | 0.2504 | 0.6339 | 0.2979 | 0.6426 |
| hybrid_sparse_rrf60 | 0.2748 | 0.6926 | 0.3175 | 0.7197 |
| hybrid_sparse_tuned | 0.2804 | 0.6866 | 0.3189 | 0.7158 |
| hybrid_best | 0.2804 | 0.6866 | 0.3189 | 0.7158 |

- **Query instruction:** with the bge instruction beats no instruction on train (0.2261 vs 0.2069 nDCG@10, 0.6152 vs 0.5845 R@100), so `dense_best` = `dense_bge_small`.
- **Dense vs sparse:** dense alone (0.2261 train) is roughly 2× baseline BM25 but slightly below the tuned `sparse_best` on train nDCG@10 (0.2435). It is ahead on train R@100 (0.6152 vs 0.5995). On dev, dense is ahead on nDCG@10 and behind on R@100.
- **Fusion:** fusing with baseline BM25 needs a heavy dense weight (tuned w_dense 0.8, k 100) and adds +0.024 train nDCG@10 over dense alone. Fusing with `sparse_best` is much better (+0.054 over dense, +0.037 over sparse_best), with equal weights at k = 20. Even untuned k = 60 equal-weight RRF with `sparse_best` reaches 0.2748 train.
- **hybrid_best** = `hybrid_sparse_tuned` with its setting frozen (k = 20, w_dense = 0.5). It was selected on train (0.2804 vs 0.2504 for the BM25 hybrid).

Train grid for the sparse_best hybrid (nDCG@10; rows k, columns w_dense). The surface is smooth with a broad plateau around w_dense 0.4–0.5 and k 10–60, so the choice is not a fragile optimum:

| k \ w_dense | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2435 | 0.2493 | 0.2573 | 0.2602 | 0.2647 | 0.2660 | 0.2624 | 0.2508 | 0.2432 | 0.2328 | 0.2261 |
| 2 | 0.2435 | 0.2474 | 0.2568 | 0.2628 | 0.2640 | 0.2674 | 0.2632 | 0.2549 | 0.2428 | 0.2310 | 0.2261 |
| 5 | 0.2435 | 0.2475 | 0.2554 | 0.2610 | 0.2686 | 0.2762 | 0.2640 | 0.2550 | 0.2418 | 0.2288 | 0.2261 |
| 10 | 0.2435 | 0.2489 | 0.2563 | 0.2615 | 0.2718 | 0.2790 | 0.2701 | 0.2556 | 0.2448 | 0.2307 | 0.2261 |
| 20 | 0.2435 | 0.2509 | 0.2609 | 0.2659 | 0.2734 | **0.2804** | 0.2682 | 0.2609 | 0.2493 | 0.2352 | 0.2261 |
| 30 | 0.2435 | 0.2534 | 0.2618 | 0.2672 | 0.2766 | 0.2789 | 0.2697 | 0.2635 | 0.2525 | 0.2397 | 0.2261 |
| 60 | 0.2435 | 0.2574 | 0.2665 | 0.2714 | 0.2773 | 0.2748 | 0.2689 | 0.2642 | 0.2580 | 0.2468 | 0.2261 |
| 100 | 0.2435 | 0.2598 | 0.2683 | 0.2727 | 0.2770 | 0.2730 | 0.2699 | 0.2670 | 0.2608 | 0.2496 | 0.2261 |
| 200 | 0.2435 | 0.2647 | 0.2696 | 0.2715 | 0.2760 | 0.2733 | 0.2681 | 0.2669 | 0.2621 | 0.2550 | 0.2261 |

### First-stage recall at depth 1000

For later reranking, the depth-1000 candidate lists:

| Candidates | train R@100 | train R@1000 | dev R@100 | dev R@1000 |
| --- | ---: | ---: | ---: | ---: |
| bm25 (baseline) | 0.3365 | 0.6402 | 0.4045 | 0.6599 |
| sparse_best | 0.5995 | 0.8462 | 0.6531 | 0.8775 |
| dense_best | 0.6152 | 0.8610 | 0.6259 | 0.8506 |
| hybrid bm25 + dense (k=100, w=0.8) | 0.6339 | 0.8674 | 0.6426 | 0.8745 |
| hybrid_best (k=20, w=0.5) | 0.6866 | 0.9017 | 0.7158 | 0.9227 |

### Per-domain nDCG@10 (dev)

| Domain | bm25 | sparse_best | dense_best | hybrid_best |
| --- | ---: | ---: | ---: | ---: |
| bitcoin | 0.0384 | 0.1756 | 0.1687 | 0.2154 |
| cardano | 0.1027 | 0.2373 | 0.1941 | 0.2578 |
| economics | 0.0467 | 0.1693 | 0.1528 | 0.1953 |
| genealogy | 0.1949 | 0.3485 | 0.3193 | 0.3701 |
| history | 0.0870 | 0.2456 | 0.2704 | 0.2930 |
| hsm | 0.2489 | 0.3876 | 0.3108 | 0.4547 |
| iota | 0.3309 | 0.5177 | 0.7128 | 0.6576 |
| law | 0.0574 | 0.2201 | 0.3440 | 0.3451 |
| monero | 0.0320 | 0.1323 | 0.0908 | 0.1742 |
| politics | 0.2625 | 0.3624 | 0.3397 | 0.4715 |
| quant | 0.0237 | 0.1037 | 0.0948 | 0.1000 |
| travel | 0.0442 | 0.1631 | 0.2496 | 0.2559 |
| workplace | 0.0230 | 0.2399 | 0.3463 | 0.3551 |
| **macro** | **0.1148** | **0.2541** | **0.2765** | **0.3189** |

Dense beats tuned sparse in 5 of 13 dev domains (history, iota, law, travel, workplace). Tuned sparse is ahead in the other 8, by the widest margins on hsm (+0.077), cardano (+0.043) and monero (+0.042). The two are complementary: `hybrid_best` beats both of its inputs in 11 of 13 dev domains. The exceptions are iota (3 dev queries; dense alone is better) and quant (sparse alone is better by 0.004).

## 4. Small-domain comparisons (iota, quant, law, workplace, cardano)

The base model and passage chunking were run on the 5 small domains only (162k documents). Rows for 13-domain systems are their scores restricted to these domains.

| System | train nDCG@10 | train R@100 | dev nDCG@10 | dev R@100 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.1082 | 0.3543 | 0.1075 | 0.4969 |
| sparse_best | 0.2660 | 0.5821 | 0.2637 | 0.7222 |
| dense_bge_small | 0.2276 | 0.6591 | 0.3384 | 0.6601 |
| dense_bge_small_maxp | 0.2307 | 0.6526 | 0.2858 | 0.6752 |
| dense_bge_base | 0.2313 | 0.6190 | 0.2660 | 0.7443 |
| hybrid_best | 0.2957 | 0.7076 | 0.3431 | 0.7754 |
| hybrid_sparse_maxp | 0.2794 | 0.6971 | 0.3211 | 0.7658 |
| hybrid_sparse_base | 0.2746 | 0.6735 | 0.3259 | 0.7771 |

Encoding cost on these 162k documents: bge-small first-512 1,191 s, bge-small MaxP 1,959 s (1.6×; 2.6 passages per document), bge-base first-512 3,288 s (2.8×).

- **On train (116 queries)**, neither variant beats bge-small first-512 by a meaningful margin. MaxP gains +0.003 nDCG@10 and loses 0.007 R@100. bge-base gains +0.004 nDCG@10 and loses 0.040 R@100.
- **On dev (50 queries, 3 of them in iota)**, both variants lose nDCG@10 and gain R@100 (bge-base +0.084). With this few queries the dev swings are not reliable, and they were not used for any choice.
- **Inside the hybrid** with sparse_best, using hybrid_best's frozen setting, bge-small first-512 is also the best dense input on train: 0.2957, vs 0.2794 with MaxP and 0.2746 with bge-base.
- **Conclusion:** on train there is no evidence that passage chunking or a base-size model pays for its extra cost, so the 13-domain system stays bge-small with the first 512 tokens. bge-base's dev recall gain is a hint, not a result. Checking it would need the full 13-domain bge-base encode (about 7 h on this machine) and a selection on train.

## 5. Reproduce

```
export RETECO_DATA=data/data/track1_tempo PYTHONPATH="$PWD/src"
python -m reteco.dense.bench --models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 intfloat/e5-base-v2
python -m reteco.dense.retrieve --config configs/dense_bge_small.yaml        # encodes on first run (~2.4 h), resumable
python -m reteco.dense.retrieve --config configs/dense_bge_small_noinst.yaml # reuses the embeddings
python -m reteco.dense.retrieve --config configs/dense_best.yaml
python -m reteco.dense.retrieve --config configs/dense_bge_small_maxp.yaml   # 5 small domains
python -m reteco.dense.retrieve --config configs/dense_bge_base.yaml         # 5 small domains
# sparse_best candidates, using the sparse worktree's code (read-only):
PYTHONPATH=../reteco-sparse/src RETECO_DATA=$PWD/data/data/track1_tempo \
  python -P src/reteco/dense/sparse_candidates.py \
  --config ../reteco-sparse/configs/sparse_best.yaml --out cache/dense/candidates/sparse_best
python -m reteco.dense.fusion --config configs/dense_hybrid_rrf60.yaml
python -m reteco.dense.fusion --config configs/dense_hybrid_sparse_rrf60.yaml
python -m reteco.dense.fusion --config configs/dense_hybrid_bm25.yaml        # tunes on train
python -m reteco.dense.fusion --config configs/dense_hybrid_sparse.yaml      # tunes on train
python -m reteco.dense.fusion --config configs/dense_hybrid_best.yaml
python -m reteco.dense.fusion --config configs/dense_hybrid_sparse_maxp.yaml  # 5 small domains
python -m reteco.dense.fusion --config configs/dense_hybrid_sparse_base.yaml  # 5 small domains
```

Encoding writes finished shards of 8,192 documents to `cache/dense/emb/<config>/<domain>/shards/`, so an interrupted run resumes at the last unfinished shard. Each run appends to `results/experiments.csv`.

## Decisions made

Choices I made without asking, with the reasoning:

1. **Data path.** The corpora are not under `data/track1_tempo/` in this worktree, so everything runs with `RETECO_DATA=data/data/track1_tempo` (the symlink to the main checkout).
2. **fp16 inference** for all models. It agrees with fp32 to cosine 1.000 and is 20–25% faster.
3. **Base model = bge-base-en-v1.5** rather than e5-base-v2. Speed is identical, and staying in the bge family (same instruction, same training recipe) makes the small-vs-base comparison about model size only.
4. **Query instruction kept.** It was chosen on train over no instruction (13 domains). Queries are HTML-stripped and truncated to 512 tokens, which cuts the tail of the few very long posts.
5. **Passage settings** 256 tokens with stride 192 (64 overlap), cut at tokenizer offsets so passages are verbatim substrings. Re-tokenising a substring can occasionally give one extra token, which then gets truncated; this is negligible. MaxP scores are computed exactly with a segmented max over all passage scores, not with a FAISS top-k, because the per-document max needs every passage score.
6. **FAISS indexes are not persisted.** A flat IP index holds exactly the cached embedding matrix and builds in under a second, so saving it would only double disk use. FAISS runs single-threaded: faiss-cpu and torch each bundle libomp on macOS, and multi-threaded search segfaulted once torch was loaded. Single-threaded flat search over 200k vectors takes under 0.5 s for 800 queries.
7. **Fusion inputs at depth 1000.** Baseline BM25 candidates were re-scored at depth 1000 from the existing `cache/bm25/` indexes (load only). Their top 100 gives exactly the logged BM25 scores (train 0.1076 / 0.3365, dev 0.1148 / 0.4045); 3 train and 2 dev lists differ from `runs/*/bm25.trec` only by ties at rank 100.
8. **sparse_best candidates** come from the sparse worktree's code: its `TFIndex.load`, per-domain (k1, b), query cleaning and `search` at depth 1000, with output only under `cache/dense/candidates/sparse_best/`. Nothing in that worktree or in `cache/sparse/` was written. The top 100 reproduces the logged sparse_best (train 0.2435 nDCG@10, dev 0.2541 / 0.6531). Train R@100 is 0.5995 vs the logged 0.5992 because 28 of about 121k (query, doc) pairs differ by ties at the cutoff.
9. **sparse_best is not logged** in this branch's `results/experiments.csv`, since the sparse branch logs it and a duplicate row would only conflict on merge. Its numbers here are computed from the candidates.
10. **Fusion selection** is by train macro nDCG@10, the official metric, with ties broken by R@100. The hybrid with the higher train R@100 (untuned RRF k = 60 with sparse_best: 0.6926 vs 0.6866) therefore lost on nDCG@10. `hybrid_best` and `dense_best` are separate frozen configs, so they appear as their own rows in the CSV, duplicating `hybrid_sparse_tuned` and `dense_bge_small`.
11. **Small-domain rows** are logged with `n_domains = 5`, including the 5-domain subset scores of `dense_bge_small` and `hybrid_best`, so the comparison in section 4 reads straight from the CSV.
12. **Only `dense_best` and `hybrid_best` run files are committed** (train and dev), as requested. The other variants' run files stay local and untracked in `runs/`, and every variant's score is in the CSV.
13. **No new files under `tests/`**, because the brief limited where files may be created. The code was instead checked with ad-hoc self-tests on a synthetic corpus: resume after a simulated mid-domain interruption gives embeddings identical to a clean encode; passages are verbatim, ordered and cover every document; MaxP search matches brute force; `clean_query` strips tags and entities.

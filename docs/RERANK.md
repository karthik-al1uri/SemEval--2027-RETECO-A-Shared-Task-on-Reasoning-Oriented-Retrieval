# Second-stage reranking and the final pipeline (Sub-track 1a)

> **Naming.** The "final system" in this document is now called `final_v1` (`configs/final_v1.yaml`, `runs/{train,dev}/final_v1.trec`). Since 2026-11-11 `final` is the playground champion with fine-tuned models (`configs/final.yaml`, `docs/PLAYGROUND.md`). Rows logged before the rename keep their original names in `results/experiments.csv`.

Cross-encoder reranking of `hybrid_best`, score interpolation with the first stage, the temporal-fit rescorer re-tuned on top, near-duplicate demotion as the last stage (section 7), and one config that rebuilds the final system. Code is in `src/reteco/rerank/` and `src/reteco/pipeline.py`, configs are `configs/rerank_*.yaml` and `configs/final_v1.yaml`, and every scored variant is a row in `results/experiments.csv` (new column `latency_ms_per_query`). All numbers are macro nDCG@10 over the 13 domains unless a table says "5 small domains". **Every choice was made on train; dev was scored after each choice and never fed back.**

## Summary

| System | train nDCG@10 | dev nDCG@10 | dev w/o iota | dev R@100 | ms/query (M4, MPS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_best` (input) | 0.2804 | 0.3189 | 0.2907 | 0.7158 | |
| + MiniLM cross-encoder top 200, score only | 0.2163 | 0.2598 | 0.2322 | 0.7032 | 1053 |
| + interpolation with the first stage (`rerank_minilm_interp`) | 0.2941 | 0.3316 | 0.3044 | 0.7436 | 1053 |
| + temporal re-tuned on top | 0.2976 | 0.3303 | 0.3030 | 0.7436 | 1055 (whole pipeline) |
| + near-duplicate demotion, re-tuned on top = **`final`** | **0.3025** | **0.3403** | 0.3105 | 0.7436 | 1100 (whole pipeline) |

- **Final system: +0.0221 train, +0.0214 dev over `hybrid_best`** (dev 95% CI [+0.0088, +0.0346], Holm-corrected p = 0.006; domain-stratified bootstrap and paired randomization test, 10k each, `results/analysis_significance.csv`). Without iota the dev gain is +0.0198.
- **Near-duplicate demotion was added last** (section 7): re-tuned on the train output of the system above, it adds +0.0049 train and +0.0100 dev (+0.0075 without iota), both significant. Before it was added, the final system was +0.0172 train / +0.0114 dev over `hybrid_best`.
- **The cross-encoder alone is worse than the first stage it reranks.** Both off-the-shelf models rank these long StackExchange questions worse than the tuned sparse+dense hybrid: MiniLM alone is 0.216 train vs 0.280 for the hybrid, and bge-reranker-base alone is 0.189. All of the gain comes from **interpolating** the reranker with the first-stage score (a = 0.55, z-score): +0.0137 train, +0.0127 dev [+0.0026, +0.0234].
- **bge-reranker-base, the intended main model, lost to the 22M MiniLM** on train (0.2834 vs 0.2941 interpolated, 13 domains) while being 6× slower per pair. I checked that this is not a bug (its logits match the sentence-transformers reference exactly, and fp16 matches fp32).
- **Temporal on top of the reranker barely moves.** Re-tuned on the reranked train run it gives +0.0035 train (+0.0023 without iota), and **−0.0013 on dev (−0.0014 without iota)**, within noise (dev CI [−0.0051, +0.0019]). It is kept in the final system because it was selected on train; the ablation shows the dev cost.
- **LLM listwise reranking was skipped: there is no `.env` file, so no `ANTHROPIC_API_KEY`.** No API calls were made.

## 1. Cross-encoders

`src/reteco/rerank/cross_encoder.py` loads the model with `transformers` (`AutoModelForSequenceClassification`) on the MPS device in fp16 and keeps the raw logit (no sigmoid), so per-query normalisation sees the full range.

- **Query**: cleaned exactly like `sparse_best` (`reteco.sparse.clean.clean_query` with `code, html, urls, signature`: `<pre>` blocks, tags and attributes, URLs and sign-offs removed), then cut to 256 tokens.
- **Document**: first 4,000 characters, then the tokenizer fills the rest of the 512-token budget (`truncation="only_second"`, so the query is never cut further).
- **Input**: `hybrid_best` rebuilt at depth 1000 from the dense agent's depth-1000 sparse and dense candidates with its frozen RRF setting (k = 20, w_dense = 0.5). Its top 100 gives exactly the logged `hybrid_best` scores (train 0.2804 / 0.6866, dev 0.3189 / 0.7158). Its macro recall on train is 0.59 / 0.69 / 0.77 / 0.90 at depth 50 / 100 / 200 / 1000, so each doubling of the reranking depth adds less gold.
- **Cache and latency**: every (query, document) logit is cached under `cache/rerank/scores/<model key>/`, together with the wall time of the call that produced it (tokenisation + forward pass + device sync). Each query is scored in rank segments (0–25, 25–50, 50–100, 100–200), so the latency for depth 50, 100 or 200 is a sum of whole measured calls, and deeper runs reuse the shallower scores. Latency is per query, one query at a time, batch size 32, after a warm-up batch.

### Throughput (`python -m reteco.rerank.bench`, `cache/rerank/bench.json`)

Apple M4, 16 GB, torch 2.15.0 on MPS; 6 train queries × 50 candidates from the small domains.

| Model | Params | Precision | pairs/s | ms/query at depth 50 | fp16 vs fp32 (Spearman per query) |
| --- | ---: | --- | ---: | ---: | ---: |
| BAAI/bge-reranker-base | 278M | fp32 | 25 | 1970 | |
| BAAI/bge-reranker-base | 278M | fp16 | 33 | 1524 | 0.9998 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | fp32 | 157 | 318 | |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | fp16 | 195 | 257 | 0.9999 |

Batch size 16, 32 and 64 were within 3% of each other; 32 is used. Measured latencies in the full runs match: about 5.3 ms per pair for MiniLM and 31 ms per pair for bge-reranker-base. At that rate a full train run (1,211 queries) takes 21 min (MiniLM, depth 200), 64 min (bge-base, depth 100) or 2.1 h (bge-base, depth 200), all under the 3-hour limit.

### Depth 50 / 100 / 200 on the 5 small domains

iota, quant, law, workplace and cardano (116 train / 50 dev queries). "interp" is the interpolated score with (a, norm) tuned on these train queries. On these domains `hybrid_best` alone scores 0.2957 train / 0.3431 dev.

| Model | Depth | alone, train | interp, train | interp a | alone, dev | interp, dev | ms/query |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MiniLM | 50 | 0.2493 | 0.3193 | 0.40 z | 0.3402 | 0.3822 | 272 |
| MiniLM | 100 | 0.2428 | 0.3201 | 0.40 z | 0.3322 | 0.3845 | 544 |
| MiniLM | 200 | 0.2356 | **0.3228** | 0.40 z | 0.3259 | 0.3795 | 1087 |
| bge-reranker-base | 50 | 0.1790 | 0.2974 | 0.95 mm | 0.2555 | 0.3404 | 1596 |
| bge-reranker-base | 100 | 0.1583 | 0.2975 | 0.95 mm | 0.2301 | 0.3404 | 3196 |
| bge-reranker-base | 200 | 0.1405 | 0.2975 | 0.95 mm | 0.2351 | 0.3404 | 6396 |

- **Alone, deeper is worse** for both models: more candidates means more chances for the reranker to promote a wrong document.
- **Interpolated, MiniLM gains a little with depth** (0.3193 → 0.3228). The interpolation keeps the first-stage score as a prior, so the reranker can pull a few gold documents up from ranks 101–200 without scrambling the top (depth 200 interpolated R@100: 0.690 train, 0.816 dev, vs 0.708 / 0.775 for the first stage).
- **bge-reranker-base adds almost nothing** on these domains: the tuned weight is a = 0.95 (95% first stage) at every depth.
- **Chosen depths** (rule: best interpolated train nDCG@10, ties within 0.001 to the cheaper depth): **MiniLM 200, bge-reranker-base 50.**

A shorter query (64 instead of 256 tokens, which leaves more room for the document and keeps mostly the title) helps each reranker alone (MiniLM 0.249 → 0.258, bge 0.179 → 0.205 at depth 50) but not after interpolation (0.3193 → 0.3186, 0.2974 → 0.2976), so 256 stays (`configs/rerank_*_q64.yaml`, train only).

### Full runs, 13 domains

| System | train nDCG@10 | dev nDCG@10 | dev w/o iota | train R@100 | dev R@100 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_best` | 0.2804 | 0.3189 | 0.2907 | 0.6866 | 0.7158 | |
| `rerank_bge_base` (depth 50, score only) | 0.1891 | 0.2253 | 0.1995 | 0.6866 | 0.7158 | 1578 |
| `rerank_bge_base_interp` (a = 0.70, z) | 0.2834 | 0.3266 | 0.2969 | 0.6866 | 0.7158 | 1578 |
| `rerank_minilm_d50` | 0.2326 | 0.2770 | 0.2465 | 0.6866 | 0.7158 | 262 |
| `rerank_minilm_d50_interp` (a = 0.60, z) | 0.2912 | 0.3305 | 0.3032 | 0.6866 | 0.7158 | 262 |
| `rerank_minilm_d100` | 0.2241 | 0.2674 | 0.2387 | 0.6866 | 0.7158 | 525 |
| `rerank_minilm_d100_interp` (a = 0.50, mm) | 0.2937 | 0.3373 | 0.3029 | 0.6866 | 0.7158 | 525 |
| `rerank_minilm` (depth 200, score only) | 0.2163 | 0.2598 | 0.2322 | 0.6545 | 0.7032 | 1053 |
| **`rerank_minilm_interp`** (depth 200, a = 0.55, z) | **0.2941** | 0.3316 | 0.3044 | 0.6984 | 0.7436 | 1053 |

The depth chosen on the small domains is also the best on all 13 train domains (0.2941 vs 0.2937 at 100 and 0.2912 at 50), though the three are within 0.003 of each other. On dev, depth 100 happens to be best (0.3373); that was not used. Depth 50 gives most of the gain at a quarter of the latency (0.2912 train, 262 ms), which is the setting to use if latency matters.

On 13 domains bge-reranker-base is more useful than on the 5 small ones (tuned a = 0.70, +0.0030 train, +0.0077 dev), but the train gain is not significant (CI [−0.0053, +0.0116]) and it is well below MiniLM.

**Why the rerankers are weak here (a hypothesis, not tested).** Both are trained for short search queries and a passage that answers them (MS MARCO for MiniLM; a broader multilingual mix for bge-reranker-base, which is XLM-R based). Our queries are 100–200-word StackExchange posts (`docs/DATA.md`), and the gold documents are passages that support an answer rather than answer the question directly. Alone, MiniLM beats the hybrid on dev only in law (0.43 vs 0.35), monero (0.20 vs 0.17) and quant (0.11 vs 0.10). It is far behind where the hybrid is strongest: politics (0.33 vs 0.47), hsm (0.25 vs 0.45) and genealogy (0.21 vs 0.37).

## 2. Score interpolation (`interpolate.py`)

```
final(q, d) = a * norm(first_stage(q, d)) + (1 - a) * norm(reranker(q, d))      over the reranked top `depth`
```

`norm` is a per-query min-max or z-score; documents below the reranking depth keep their first-stage order underneath. Grid on train: a ∈ {0, 0.05, …, 1} × norm ∈ {minmax, zscore}; ties go to the smaller a. The first-stage score is the RRF score of `hybrid_best`.

Train macro nDCG@10 of `rerank_minilm` (depth 200) over a:

| a | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.45 | 0.5 | **0.55** | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minmax | 0.2163 | 0.2459 | 0.2678 | 0.2790 | 0.2861 | 0.2914 | 0.2939 | 0.2923 | 0.2912 | 0.2895 | 0.2859 | 0.2823 | 0.2804 |
| zscore | 0.2163 | 0.2484 | 0.2713 | 0.2807 | 0.2904 | 0.2916 | 0.2918 | **0.2941** | 0.2930 | 0.2890 | 0.2862 | 0.2820 | 0.2804 |

The surface is smooth with a plateau from a = 0.45 to 0.65 (all within 0.003 of the best), so the choice is not a fragile optimum. Any a between 0.3 and 0.9 beats both inputs. Full grids: `cache/rerank/interp/<system>_grid_train.csv`.

Interpolation also raises Recall@100 (train 0.6866 → 0.6984, dev 0.7158 → 0.7436), because reranking the top 200 pulls some gold documents from ranks 101–200 into the top 100.

## 3. Temporal rescorer on top

`python -m reteco.temporal.tune --run runs/train/rerank_minilm_interp.trec --prefix rerank_temporal_` re-tunes the full grid of `docs/TEMPORAL.md` on the reranked train run (the new `--prefix` option writes `configs/rerank_temporal_*.yaml` and keeps the BM25-tuned `configs/temporal_*.yaml`). Document times for the new candidates were extracted first (`reteco.temporal.doctimes`, under 30 s). The two gated variants were not re-run, since the gate added nothing on BM25.

| Variant | Selected on train | train | Δ train | Δ train w/o iota | dev | Δ dev | Δ dev w/o iota |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (none: `rerank_minilm_interp`) | | 0.2941 | | | 0.3316 | | |
| `boost` | τ 25, w (2,1,0), center 0, minmax, b 0.1 | 0.2953 | +0.0013 | +0.0014 | 0.3309 | −0.0006 | −0.0007 |
| `fit` | same as `boost` (center 0 won) | 0.2953 | +0.0013 | +0.0014 | 0.3309 | −0.0006 | −0.0007 |
| `fit_years` | τ 25, w (1,2,0), center 0, minmax, b 0.1 | 0.2957 | +0.0016 | +0.0017 | 0.3301 | −0.0015 | −0.0016 |
| **`fit_present`** | τ 25, w (1,0,0), center 0, minmax, b 0.1 | **0.2976** | **+0.0035** | +0.0023 | 0.3303 | −0.0013 | −0.0014 |

- **Gain with iota: +0.0035 train, −0.0013 dev. Without iota: +0.0023 train, −0.0014 dev.** Train CI [+0.0010, +0.0064]; dev CI [−0.0051, +0.0019]. On dev, 56 queries improve, 35 get worse and 428 are unchanged.
- **Little temporal signal is left after reranking.** On BM25 the rescorer added +0.0095 train. On the reranked run the gain is a third of that, the chosen weight is small (b = 0.1 on min-max scores, so the temporal term moves a document by at most a tenth of the query's score range), and the b curve is sharply peaked: 0.2956 at b = 0.05, 0.2976 at 0.1, 0.2938 at 0.2 and falling fast after that (0.2505 at b = 1). A plausible reason: the first stage and the cross-encoder both see the dates in the query and the document, so wrong-period documents are already lower.
- **The optimum flipped from penalise to boost** (center 0 instead of 1) and every variant picks the same τ = 25 and b = 0.1. With candidates that are already time-consistent, a small push for the closest time match helps on train more than pushing outliers down.
- **By domain (dev)** the rescorer still helps where it did on BM25: history +0.003, genealogy +0.004, monero +0.006. It loses in law (−0.016, 11 queries) and travel (−0.016, where few queries carry a time and a boosted date mention is usually noise).

## 4. LLM listwise reranking: skipped

The brief made this optional and conditional on `ANTHROPIC_API_KEY` in `.env`. There is no `.env` in the repository, so it was not attempted: no code, no API calls, no cost. With a key, the plan was: top 20 of `rerank_minilm_interp` for the small domains only, a listwise prompt asking whether each passage fits the query's time period and topic, every response cached under `cache/llm/`, the cost estimate printed first and a $10 cap.

## 5. Final pipeline (`src/reteco/pipeline.py`, `configs/final_v1.yaml`)

```
python -m reteco.pipeline --config configs/final_v1.yaml              # runs/{train,dev}/final_v1.trec, logged
python -m reteco.pipeline --config configs/final_v1.yaml --ablation   # + results/ablation.csv (dev)
```

Stages, each recomputed on every call: sparse BM25 (`configs/sparse_best.yaml`, depth 1000) → dense bge-small (`configs/dense_best.yaml`, depth 1000) → RRF (k 20, w_dense 0.5) → MiniLM cross-encoder on the top 200 → interpolation (a 0.55, z-score) → temporal (`configs/rerank_temporal_fit_present.yaml`) → near-duplicate demotion (`configs/final_dedup_end.yaml`, section 7). Only input-keyed caches are reused, and each is rebuilt when missing: sparse term-count indexes, document embeddings, cross-encoder pair scores and document time extractions. From an empty `cache/` the same command rebuilds everything (about 2.4 h of embedding and 30 min of cross-encoder scoring on this machine, plus a few minutes of indexing).

The pipeline's runs match the stage-by-stage runs exactly (train 0.3025, dev 0.3403; before the dedup stage was added, 0.2976 and 0.3303). Its sparse and dense stages, recomputed from the indexes and embeddings, give the same depth-1000 candidates as the cached ones, so no new cross-encoder pairs were needed.

Latency per query (dev, ms; first stage with indexes and embeddings in memory, one process, M4):

| sparse | dense (query encoding + exact search) | fusion | cross-encoder (top 200) | interpolation | temporal | dedup | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.2 | 11.7 | 1.8 | 1052.5 | 0.1 | 2.8 | 26.7 | 1100 |

The sparse, dense, temporal and dedup stages are batched over all queries and divided by the number of queries; the cross-encoder is measured one query at a time. 96% of the time is the cross-encoder. Depth 50 would bring the total to about 310 ms at −0.003 train nDCG@10 (measured before the dedup stage was added).

## 6. Ablation (dev, `results/ablation.csv`)

Each row removes one component from `final` and keeps every other setting fixed (decision 10). Scored once on dev; nothing here was used for a choice.

| Removed | dev nDCG@10 | Δ vs final | dev w/o iota | Δ w/o iota | dev R@100 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nothing (`final`) | 0.3403 |  | 0.3105 |  | 0.7436 | 1100 |
| sparse retrieval (first stage = dense only) (`no_sparse`) | 0.3082 | -0.0321 | 0.2700 | -0.0405 | 0.6641 | 1085 |
| dense retrieval (first stage = sparse only) (`no_dense`) | 0.2963 | -0.0440 | 0.2720 | -0.0385 | 0.7019 | 1102 |
| cross-encoder (and so interpolation): fusion -> temporal -> dedup (`no_rerank`) | 0.3263 | -0.0140 | 0.2953 | -0.0152 | 0.7158 | 55 |
| score interpolation (a = 0: cross-encoder score only) (`no_interp`) | 0.2644 | -0.0759 | 0.2372 | -0.0733 | 0.7032 | 1095 |
| temporal rescorer (`no_temporal`) | 0.3406 | +0.0003 | 0.3107 | +0.0002 | 0.7436 | 1097 |
| query cleaning for the cross-encoder (raw HTML query) (`no_query_clean`) | 0.3348 | -0.0055 | 0.3013 | -0.0092 | 0.7261 | 1112 |
| near-duplicate demotion (`no_dedup`) | 0.3303 | -0.0100 | 0.3030 | -0.0075 | 0.7436 | 1073 |

The ablation was re-run after near-duplicate demotion became part of `final`, so every row includes the dedup stage except `no_dedup` (which equals the previous `final`).

- **Interpolation matters most** (−0.076): without it the system is the cross-encoder's own ranking, which is far below the first stage (section 1).
- **Both first-stage retrievers are needed** (−0.044 without dense, −0.032 without sparse; about −0.04 each without iota). Each alone loses recall (R@100 0.70 and 0.66 vs 0.74), and the reranker cannot recover documents the first stage missed.
- **The cross-encoder is worth +0.014** (fusion → temporal → dedup gives 0.3263) at 20× the latency (55 ms → 1100 ms per query).
- **Near-duplicate demotion is worth +0.010** (+0.0075 without iota) for 27 ms per query.
- **Query cleaning for the cross-encoder is worth +0.006** (+0.009 without iota): feeding it the raw HTML post costs nDCG and recall.
- **Temporal is the only component whose removal helps on dev**, now by only +0.0003 (+0.0002 without iota; it was +0.0013 before the dedup stage). It was kept because it was selected on train (decision 7).

Per-domain dev scores for every variant are in `results/ablation.csv`; each variant is also a row in `results/experiments.csv` (`final_v1_no_*`, logged as `final_no_*` before the rename).

## 7. Near-duplicate demotion on the final system

The analysis branch's rescorer (`src/reteco/analysis_tools/dedup.py`, `docs/ANALYSIS.md` sections 7 and 8) demotes a document that repeats a higher-ranked document of the same list (identical up to case and punctuation, or word 5-gram Jaccard ≥ threshold). On `hybrid_best` both variants helped on train. Both were re-tuned on `runs/train/final.trec` (the system above, before demotion), and the variant with the higher train nDCG@10 was adopted:

| Variant (re-tuned on final train) | Threshold | train nDCG@10 | Δ train |
| --- | --- | ---: | ---: |
| (none: final before demotion) | | 0.2976 | |
| `dedup`: one step down (skip 1) | Jaccard ≥ 0.8 | 0.3001 | +0.0026 |
| **`dedup_end`: to the end of the list (skip 0)** | Jaccard ≥ 0.85 | **0.3025** | **+0.0049** |

`dedup_end` was adopted (`configs/final_dedup_end.yaml`) and scored once on dev as `final+dedup`: **0.3303 → 0.3403 dev (+0.0100, CI [+0.0041, +0.0170]), 0.3030 → 0.3105 without iota.** It is now stage 7 of the pipeline, so `final` = `final+dedup` (identical runs on both splits). The dedup stage costs about 27 ms per query (shingling and pairwise Jaccard over the top 100; reading the document texts is counted as document-side work, like the indexes). Details, the threshold grid and the gold audit: `docs/ANALYSIS.md` section 8.

## 8. Reproduce

```
source .venv-dense/bin/activate                  # Python 3.13, requirements.txt + requirements-dense.txt
export PYTHONPATH="$PWD/src"
python -m reteco.rerank.bench                                                   # throughput
for d in 200 100 50; do for m in minilm bge_base; do
  python -m reteco.rerank.run --config configs/rerank_$m.yaml --domains small --depth $d
done; done                                                                      # depth comparison, 5 small domains
python -m reteco.rerank.run --config configs/rerank_minilm.yaml                 # depth 200, 13 domains (~30 min)
python -m reteco.rerank.run --config configs/rerank_minilm.yaml --depth 50
python -m reteco.rerank.run --config configs/rerank_minilm.yaml --depth 100
python -m reteco.rerank.run --config configs/rerank_bge_base.yaml               # depth 50 (~45 min)
python -m reteco.temporal.doctimes --runs runs/train/rerank_minilm_interp.trec runs/dev/rerank_minilm_interp.trec
python -m reteco.temporal.tune --run runs/train/rerank_minilm_interp.trec --prefix rerank_temporal_ \
    --variants boost fit fit_years fit_present
for v in boost fit fit_years fit_present; do for s in train dev; do
  python -m reteco.temporal.rescore --run runs/$s/rerank_minilm_interp.trec --split $s --config configs/rerank_temporal_$v.yaml
done; done
# near-duplicate demotion (section 7) on the final system before demotion; the base is the
# temporal run above, since runs/{train,dev}/final_v1.trec already include demotion
B=rerank_minilm_interp+rerank_temporal_fit_present
python -m reteco.analysis_tools.dedup --tune --skip 0 --run runs/train/$B.trec \
    --config /tmp/final_dedup_end.yaml --grid-out results/analysis_dedup_grid_final_train.csv
                        # grid only; the committed configs/final_dedup*.yaml are not touched
for s in train dev; do
  python -m reteco.analysis_tools.dedup --run runs/$s/$B.trec --split $s \
    --config configs/final_dedup_end.yaml --out runs/$s/$B+dedup.trec --no-log
done
python scripts/analysis_dedup.py --base $B --variant dedup:final_dedup_end --tag _final
python -m reteco.pipeline --config configs/final_v1.yaml             # final system: runs/{train,dev}/final_v1.trec
python -m reteco.pipeline --config configs/final_v1.yaml --ablation-only   # results/ablation.csv (dev)
python -m pytest
```

Long runs were started with `nohup caffeinate -i …` so that they survive a terminal restart and the machine does not sleep. Everything resumes from the score cache: one interrupted run (the editor restart killed the first full MiniLM run while it was scoring history) continued where it stopped, and only complete pairs are ever logged.

## Decisions made

Choices made without asking. Each is easy to revisit.

1. **Python environment.** A new Python 3.13 venv `.venv-dense/` in this checkout with `requirements.txt` + `requirements-dense.txt`, installed with the same versions as `reteco-dense/.venv` (torch 2.15.0, transformers 5.22.2, sentence-transformers 6.2.0). The dense worktree's venv lacks `dateparser` and `scikit-learn`, which the temporal module needs, and installing into another worktree's venv seemed worse than making one here. `dateparser` and `scikit-learn` were also missing from `requirements.txt` and are now listed.
2. **Raw `transformers` instead of `sentence_transformers.CrossEncoder`**, to control truncation and timing exactly. Its logits were checked against `CrossEncoder.predict` (identical to 3 decimals).
3. **Truncation.** Query cut to 256 tokens, then only the document is truncated to fit 512. The library default (`longest_first`) would cut long queries and documents alike. 64-token queries were tried on train (section 1) and did not help.
4. **Depth per model**, not one shared depth: each model got the depth that was best for it on small-domain train, with near-ties going to the cheaper depth. MiniLM 200 (+0.0027 over 100) and bge-reranker-base 50 (all three depths within 0.0001). bge at depth 50 also means its full run took 45 min instead of 2 h, for a model that adds almost nothing at any depth.
5. **Reranker selection** on 13-domain train interpolated nDCG@10: MiniLM (0.2941) over bge-reranker-base (0.2834), even though bge was named as the main model.
6. **Interpolation uses the RRF score** of `hybrid_best` as the first-stage score (not the rank or the separate sparse and dense scores), with a per-query normalisation chosen on train together with a.
7. **Temporal stays in the final system** although it is slightly negative on dev. Train selected it (+0.0035, CI above 0), and removing it because of dev would be tuning on dev. The ablation reports what it costs.
8. **Temporal variants re-tuned**: `boost`, `fit`, `fit_years`, `fit_present`. The two gated variants were skipped (on BM25 the gate changed nothing and the classifier adds a dependency on `guidance_train.jsonl`).
9. **Latency column.** `results/experiments.csv` gained a last column `latency_ms_per_query`, blank for all earlier rows. For rerank rows it is the cross-encoder time per query (interpolation adds < 0.1 ms). For temporal rows on a reranked run it is that plus the rescoring time (the rescorer reads the input run's latency from its `.meta.json`). For pipeline rows it is the sum of all stages. Document-side work (indexes, embeddings, time extraction) is done offline and not counted. `reteco.temporal.rescore` now records this; nothing else about its output changed.
10. **Ablation keeps every other setting fixed**: removing a component does not re-tune the rest (for example, `no_rerank` applies the temporal config tuned on the reranked run directly to the fusion output; `no_sparse` interpolates with the dense cosine score instead of the RRF score). That measures each component's contribution to this system, not the best system without it.
11. **Small-domain rows** are logged with `n_domains = 5` and names ending in `_small`, so they cannot be confused with 13-domain rows.
12. **Run files.** Only `runs/{train,dev}/final_v1.trec` and their `.meta.json` are committed. Reranker, temporal and ablation variants are gitignored (`runs/*/rerank_*`, `runs/*/final_*`, `runs/*/*+rerank_temporal_*`), and every score is in the CSV.
13. **A bug found by the smoke test and fixed before any full run**: the pipeline reused the loaded cross-encoder across ablation variants, and the encoder applied its own query-cleaning setting, so the first small-domain `no_query_clean` run silently used cleaned queries. Query handling now comes from each call's config, the encoder is only reused when model, precision and length match, the polluted cache was deleted, and `tests/test_rerank.py` has a regression test. No logged number was affected (`rerank.run` uses one config per process, and the smoke test was not logged).
14. **No LLM reranking** (no API key; section 4).
15. **Near-duplicate demotion adopted as the last stage** (section 7). Both variants were re-tuned on the final system's train output and the one with the higher train nDCG@10 (`dedup_end`, +0.0049) was adopted, with the adoption rule fixed before dev was scored. The `configs/dedup*.yaml` files tuned on `hybrid_best` are kept unchanged; the re-tuned ones are `configs/final_dedup*.yaml`. The ablation was re-run with dedup in the system, so `results/ablation.csv` has a `no_dedup` row and its other rows changed slightly. `final+dedup` run files are gitignored like the other variants (`runs/*/final+*`).

# Sparse retrieval: tuned BM25, RM3, corpus and query cleaning

Sub-track 1a, all 13 domains, data v1.1. Everything was tuned on **train** only; dev was scored once per variant and never used for selection. Every scored run is in `results/experiments.csv`. Runs are in `runs/<split>/<system>.trec`, configs in `configs/sparse_*.yaml`, and code in `src/reteco/sparse/`.

## Recommendation

Use **`sparse_best`** (`configs/sparse_best.yaml`, identical to `sparse_qclean_tuned`):
- BM25 on the raw corpus
- queries cleaned of `<pre>` code blocks, HTML tags and attributes, URLs and sign-offs
- per-domain (k1, b) tuned on train
- no RM3, no corpus cleaning

| Macro | nDCG@10 train | nDCG@10 dev | R@100 train | R@100 dev |
| --- | ---: | ---: | ---: | ---: |
| BM25 baseline (`bm25`) | 0.1076 | 0.1148 | 0.3365 | 0.4045 |
| **`sparse_best`** | **0.2435** | **0.2541** | **0.5992** | **0.6531** |

That is about 2.2× the baseline nDCG@10 on dev and +0.25 Recall@100 on dev. Runs: `runs/train/sparse_best.trec`, `runs/dev/sparse_best.trec` (top 100 for every query).

Almost all of the gain comes from two changes: (1) much larger k1 and b than the official 0.9/0.4, and (2) stripping HTML from the queries. RM3 and corpus cleaning both hurt nDCG@10 on train and were not adopted.

## All variants (macro over 13 domains)

Each row changes one thing relative to the row named in "vs". Δ is nDCG@10 relative to that row.

| System | Change | vs | nDCG@10 train | Δ train | nDCG@10 dev | Δ dev | R@100 train | R@100 dev |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | official baseline, k1=0.9 b=0.4 | | 0.1076 | | 0.1148 | | 0.3365 | 0.4045 |
| `sparse_bm25_global` | one global (k1, b) = (3.0, 0.8) | bm25 | 0.1754 | +0.068 | 0.2070 | +0.092 | 0.5239 | 0.5840 |
| `sparse_bm25_domain` | per-domain (k1, b) | bm25 | 0.2220 | +0.114 | 0.2305 | +0.116 | 0.5761 | 0.6303 |
| `sparse_rm3` | + RM3 (3 docs, 80 terms, α=0.9) | bm25_domain | 0.2106 | −0.011 | 0.2217 | −0.009 | 0.5812 | 0.6401 |
| `sparse_corpus_markup` | corpus: markup + boilerplate stripped | bm25_domain | 0.2187 | −0.003 | 0.2276 | −0.003 | 0.5765 | 0.6297 |
| `sparse_corpus_clean` | corpus: markup + repeated sentences | bm25_domain | 0.2178 | −0.004 | 0.2271 | −0.003 | 0.5769 | 0.6300 |
| `sparse_query_html` | queries: HTML stripped | bm25_domain | 0.2372 | +0.015 | 0.2509 | +0.020 | 0.5967 | 0.6556 |
| `sparse_query_code` | queries: + `<pre>` blocks dropped | query_html | 0.2370 | −0.000 | 0.2518 | +0.001 | 0.5999 | 0.6556 |
| `sparse_query_full` | queries: + URLs, sign-offs dropped | query_code | 0.2377 | +0.001 | 0.2523 | +0.001 | 0.6005 | 0.6556 |
| `sparse_qclean_tuned` | (k1, b) retuned for cleaned queries | query_full | **0.2435** | +0.006 | **0.2541** | +0.002 | 0.5992 | 0.6531 |
| `sparse_qclean_markup_tuned` | + markup corpus (k1, b retuned) | qclean_tuned | 0.2431 | −0.000 | 0.2470 | −0.007 | 0.5984 | 0.6494 |
| `sparse_qclean_rm3` | + RM3 (3 docs, 80 terms, α=0.9) | qclean_tuned | 0.2380 | −0.005 | 0.2467 | −0.007 | 0.6069 | 0.6465 |
| `sparse_best` | copy of `sparse_qclean_tuned` | | 0.2435 | | 0.2541 | | 0.5992 | 0.6531 |

## 1. Tuning k1 and b

**Setup.** `src/reteco/sparse/index.py` keeps raw term counts per domain (under `cache/sparse/tf/`; `cache/bm25/` is untouched), so BM25 weights can be recomputed for any (k1, b). Scoring is exactly the baseline's (gensim `LuceneBM25Model`, query weighted too). With (0.9, 0.4) it reproduces the baseline's per-domain nDCG@10 and R@100 exactly. Grid: k1 ∈ {0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2, 3, 4, 6, 8, 12} × b ∈ {0.1, …, 1.0}, scored on all train queries. Selection maximizes nDCG@10: macro over domains for the global setting, the domain mean for per-domain settings.

**Larger k1 and b are much better everywhere.** The official (0.9, 0.4) is far from optimal on every domain. The best global setting on train is (k1=3.0, b=0.8). It gives +0.068 nDCG@10 on train, +0.092 on dev and +0.19 Recall@100 on train. Per-domain optima are mostly k1 = 6–12. A plausible reason, not tested separately: in this BM25 variant k1 also saturates the query side, and queries are long posts (≈140 tokens) that repeat their key terms. A large k1 keeps query term weights close to linear in query tf, and a large b penalizes the long, off-topic documents (code dumps, spam pages) that match many of the query's words.

**Per-domain vs global.** Per-domain is better on train, in-sample as expected (0.2220 vs 0.1754). To check that this is not just overfitting, I ran 5-fold cross-validation within train (folds over queries, seed 13). Both strategies were selected on 4/5 of the queries and scored on the held-out 1/5:

| Held-out train nDCG@10 (5-fold CV) | global | per-domain |
| --- | ---: | ---: |
| raw queries | 0.1742 | 0.2104 |
| cleaned queries | 0.1840 | 0.2312 |

**Answer on dev: yes, per-domain tuning beats one global setting:** 0.2305 vs 0.2070 nDCG@10 and 0.630 vs 0.584 Recall@100. It is better in 11 of 13 domains, ties on iota (the same setting was chosen) and is worse only on bitcoin (0.151 vs 0.157). The gap is largest where the domain optimum is far from the global one: hsm (0.339 vs 0.231), economics (0.178 vs 0.120), politics (0.327 vs 0.284).

Selected settings for `sparse_best` (cleaned queries):

| bitcoin | cardano | economics | genealogy | history | hsm | iota | law | monero | politics | quant | travel | workplace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 / 0.9 | 12 / 0.6 | 8 / 0.7 | 4 / 0.6 | 6 / 0.8 | 12 / 0.3 | 8 / 0.5 | 12 / 0.9 | 8 / 1.0 | 8 / 0.3 | 6 / 1.0 | 6 / 1.0 | 3 / 0.7 |

(k1 / b. Global fallback: k1=8, b=0.7.)

## 2. Pseudo-relevance feedback (RM3)

`src/reteco/sparse/retrieve.py` implements RM3 the way Anserini does:
- relevance model over the top `fb_docs` BM25 documents, P(w|R) ∝ Σ_d s(d)·tf(w,d)/|d|, with BM25 scores normalized to sum 1
- candidate terms must match `[a-z0-9]{2,20}` and appear in at most 10% of the domain's documents
- keep the top `fb_terms` terms
- interpolate with the L1-normalized original query weights: α·q + (1−α)·P(·|R). α=1 gives back plain BM25.

Grid on train: fb_docs ∈ {3, 5, 10, 20} × fb_terms ∈ {10, 20, 40, 80} × α ∈ {0.3, 0.5, 0.7, 0.9}, one global setting, on top of the tuned per-domain BM25.

**Result: RM3 lowers nDCG@10 at every one of the 64 grid points**, both on raw queries (best 0.2106 vs 0.2220 without RM3) and on cleaned queries (0.2380 vs 0.2435). The best settings all use α=0.9 and 3 feedback documents, the smallest amount of feedback in the grid, so the trend points to "no RM3". It does raise Recall@100 a little: +0.005 to +0.012 on train and +0.010 on dev on raw queries; on cleaned queries +0.008 on train but −0.007 on dev. Only 2 of 13 domains gain nDCG@10 on train (cardano, travel). The queries are already long (100+ terms after cleaning), and expansion mostly adds drift, which matches the usual finding that RM3 helps short keyword queries.

## 3. Corpus cleaning

`src/reteco/sparse/clean.py`, two variants, each a separate index under `cache/sparse/tf/`:

- **`markup`** strips HTML tags and entities, BBCode (`[url=…]`, `[b]`; about 31% of cardano documents are spam with BBCode), Markdown emphasis, image and link syntax, URLs and e-mail addresses. It also drops sentences that match software license headers (MIT, BSD, Apache, GPL, "Use of this source code is governed by…"), open-access and copyright footers, financial and legal disclaimers, cookie, newsletter and nav text, "Press enter or click to view image in full size", "Full text not available from this repository", and citation metadata ("Retrieved 18 January 2014", "Archived from the original on…").
- **`clean`** is `markup` plus removal of "repeated footer" sentences: any sentence of ≥5 words that occurs in ≥20 documents of the same domain. The lists are computed from the documents alone and cached in `cache/sparse/boilerplate/`. They range from 10 sentences (cardano) to 2,126 (history).

The cleaning removes 0.4–7% of tokens (largest in travel 6.8%, cardano 6.1%, bitcoin 5.5%).

**Result: no gain.** With the (k1, b) tuned on the raw corpus, `markup` loses 0.003 nDCG@10 on train and dev, and `clean` loses 0.004 / 0.003. Recall@100 is flat. After retuning (k1, b) for the markup corpus with cleaned queries, it still ties on train (0.2431 vs 0.2435; CV 0.2279 vs 0.2312) and loses 0.007 on dev. My reading: the passages are already mostly clean prose, BM25's idf already discounts boilerplate terms, and much of what the repeated-sentence filter catches is not boilerplate at all but content shared by near-duplicate passages (iota glossary entries, history book-index lines), so removing it can hurt. Not adopted.

## 4. Query cleaning

Queries are Stack Exchange posts: the title plus an HTML body. 55% contain links, 22% blockquotes, 8% images and 1.4% `<pre>` code blocks. Without cleaning, the analyzer indexes tag and attribute tokens (`p`, `href`, `rel`, `nofollow`, `noreferrer`, URL pieces). Steps, added one at a time:

| Step | mean query tokens (train) | nDCG@10 train | nDCG@10 dev |
| --- | ---: | ---: | ---: |
| none | 141.0 | 0.2220 | 0.2305 |
| `html`: drop tags (with attributes), unescape entities | 111.2 | 0.2372 | 0.2509 |
| + `code`: drop `<pre>` blocks (inline `<code>` kept) | 110.3 | 0.2370 | 0.2518 |
| + `urls`: drop URLs left in the text | 108.6 | 0.2377 | 0.2523 |
| + `signature`: "Thanks in advance", "Any help appreciated", "Regards, …", "EDIT:" | 108.5 | 0.2377 | 0.2523 |
| all four, (k1, b) retuned | 108.5 | **0.2435** | **0.2541** |

Stripping HTML is the second-largest gain in this study (+0.015 train, +0.020 dev). The other steps change little, because code blocks and sign-offs are rare. All four steps are kept since none hurt on train. Retuning (k1, b) on the cleaned queries adds another +0.006 on train.

## Per-domain results (dev, nDCG@10 / Recall@100)

| Domain | bm25 | bm25_global | bm25_domain | query_full | rm3 | **sparse_best** |
| --- | --- | --- | --- | --- | --- | --- |
| bitcoin | 0.038 / 0.282 | 0.157 / 0.515 | 0.151 / 0.521 | 0.182 / 0.538 | 0.140 / 0.534 | 0.176 / 0.538 |
| cardano | 0.103 / 0.456 | 0.094 / 0.738 | 0.117 / 0.656 | 0.146 / 0.667 | 0.116 / 0.656 | 0.237 / 0.687 |
| economics | 0.047 / 0.222 | 0.120 / 0.448 | 0.178 / 0.447 | 0.173 / 0.485 | 0.165 / 0.455 | 0.169 / 0.502 |
| genealogy | 0.195 / 0.387 | 0.297 / 0.633 | 0.328 / 0.709 | 0.353 / 0.693 | 0.288 / 0.688 | 0.349 / 0.665 |
| history | 0.087 / 0.215 | 0.174 / 0.530 | 0.182 / 0.550 | 0.246 / 0.624 | 0.169 / 0.541 | 0.246 / 0.624 |
| hsm | 0.249 / 0.593 | 0.231 / 0.582 | 0.339 / 0.632 | 0.388 / 0.658 | 0.318 / 0.630 | 0.388 / 0.658 |
| iota | 0.331 / 0.917 | 0.514 / 1.000 | 0.514 / 1.000 | 0.532 / 1.000 | 0.523 / 1.000 | 0.518 / 1.000 |
| law | 0.057 / 0.352 | 0.242 / 0.618 | 0.250 / 0.677 | 0.250 / 0.655 | 0.251 / 0.695 | 0.220 / 0.655 |
| monero | 0.032 / 0.175 | 0.124 / 0.430 | 0.133 / 0.443 | 0.132 / 0.496 | 0.143 / 0.443 | 0.132 / 0.496 |
| politics | 0.262 / 0.544 | 0.284 / 0.591 | 0.327 / 0.623 | 0.362 / 0.655 | 0.304 / 0.627 | 0.362 / 0.655 |
| quant | 0.024 / 0.200 | 0.079 / 0.258 | 0.092 / 0.592 | 0.104 / 0.692 | 0.092 / 0.642 | 0.104 / 0.642 |
| travel | 0.044 / 0.356 | 0.144 / 0.594 | 0.148 / 0.717 | 0.172 / 0.733 | 0.144 / 0.725 | 0.163 / 0.742 |
| workplace | 0.023 / 0.561 | 0.231 / 0.655 | 0.239 / 0.627 | 0.240 / 0.627 | 0.228 / 0.685 | 0.240 / 0.627 |
| **macro** | **0.1148** / 0.4045 | **0.2070** / 0.5840 | **0.2305** / 0.6303 | **0.2523** / 0.6556 | **0.2217** / 0.6401 | **0.2541** / 0.6531 |

Dev has only 3 queries for iota, 10 for quant and 11 each for law and workplace, so per-domain dev numbers for those domains are noisy. `sparse_best` and `sparse_query_full` differ only in the retuned (k1, b) and are within noise of each other on dev.

## Notes for the rerankers

- Recall@100 of `sparse_best` is 0.60 train / 0.65 dev, up from 0.34 / 0.40. About a third of gold documents are still outside the top 100.
- If a reranker benefits from slightly more recall at the cost of first-stage precision, `sparse_rm3` has the highest dev Recall@100 among the raw-query systems (0.640 vs 0.630) but is worse on nDCG@10. With cleaned queries RM3's recall gain did not carry over to dev (0.6465 vs 0.6531), so it is not recommended.
- The same `sparse.run` code can write deeper runs (`depth` in the config) if a reranker wants top 200 or more.

## Reproduce

```
export RETECO_DATA=data/data/track1_tempo     # full corpora
export PYTHONPATH="$PWD/src"

python -m reteco.sparse.build --corpus raw markup clean          # indexes -> cache/sparse/tf/
python -m reteco.sparse.tune bm25 --config configs/sparse_bm25_domain.yaml --write per_domain
python -m reteco.sparse.tune rm3  --config configs/sparse_rm3.yaml --write rm3
python -m reteco.sparse.run --config configs/sparse_best.yaml --splits train dev
```

On a 10-core laptop with 4 worker processes, building the indexes takes about 4–8 min per corpus variant, a full (k1, b) sweep 10–25 min and an RM3 sweep 5–20 min. The ranges depend on what else is running. Everything is resumable:
- indexes are cached per domain (`cache/sparse/tf/`)
- sweep results are cached per domain and grid point (`cache/sparse/tune/`)
- finished run domains are cached per config (`cache/sparse/runs/`)
- `sparse.run` appends a (system, split, config) row to `results/experiments.csv` only once (`--relog` to force).

## Decisions made

Choices made without asking. Each is easy to revisit.

1. **Selection metric**: nDCG@10 (the official metric) for every choice. Recall@100 is reported but not optimized.
2. **Global vs per-domain on train**: per-domain tuning always wins in-sample, so I used 5-fold CV within train to decide which strategy to carry forward. Dev was only used to answer the question afterwards.
3. **"One change at a time"**: RM3, corpus cleaning and query cleaning are each measured as a single change on top of the best tuned BM25 (`sparse_bm25_domain`) with its (k1, b) fixed. Only the final candidates (`sparse_qclean_tuned`, `sparse_qclean_markup_tuned`) were retuned, so that cleaning is not penalized by parameters tuned for raw text.
4. **k1 grid**: the first grid stopped at k1=6, and many per-domain optima sat on that edge, so I extended it to 8 and 12. Three domains (cardano, hsm, law) still pick k1=12. The macro train curve is flat there (best macro nDCG@10 per k1: 0.1900 at k1=4, 0.1912 at 6, 0.1915 at 8, 0.1905 at 12, with cleaned queries), so I stopped instead of going further.
5. **RM3 parameters are global**, not per domain. RM3 already loses on macro train nDCG@10 with one setting, and per-domain RM3 selection on 7–105 queries per domain would mostly fit noise.
6. **Query weighting** stays as in the official baseline (gensim, BM25-saturated query tf). Changing it would be a separate experiment.
7. **Repeated-sentence threshold**: ≥5 words and ≥20 documents per domain, picked by inspection, not tuned (the variant did not help, so I did not tune it further).
8. **Query cleaning rules** were written after looking at the query text of both splits: pattern counts only, no labels. Query text is a legal prediction-time input, and no parameter was fitted on dev.
9. **Run files**: only `runs/{train,dev}/sparse_best.trec` (and its `.meta.json`) are committed. The other variants' runs (about 10 MB per system) are gitignored (`runs/*/sparse_*`). Their scores are all in `results/experiments.csv`, and each one can be regenerated with `python -m reteco.sparse.run --config configs/<system>.yaml`.
10. **Zero-score documents** are not padded into runs. A query with fewer than 100 matching documents would get a shorter list, but every query in every run here has 100.

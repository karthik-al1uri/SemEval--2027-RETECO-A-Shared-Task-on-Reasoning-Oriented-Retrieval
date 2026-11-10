# Analysis of the existing systems (Sub-track 1a)

Post-hoc analysis of five existing runs on **dev**: `bm25`, `bm25+temporal_fit_present`, `sparse_best`, `dense_best`, `hybrid_best`. No new retrieval system was built and no run was changed. Code is in `src/reteco/analysis_tools/` and `scripts/analysis_*.py`, outputs are in `results/analysis_*.csv`, figures are in `paper/figures/`, and the paper draft is in `paper/`.

The per-query scorer reproduces every logged macro score exactly (dev nDCG@10 / R@100: bm25 0.1148 / 0.4045, temporal 0.1304 / 0.4045, sparse 0.2541 / 0.6531, dense 0.2765 / 0.6259, hybrid 0.3189 / 0.7158).

## Summary

- **Fusion helps every query type.** Hybrid is the best of the five systems for all seven temporal types on dev. On train (guidance labels) it is best for six; the exception is queries without temporal intent, where dense alone is better.
- **Sparse and dense win different types, on both splits.** Tuned sparse is ahead on *specific date*, *ordering/sequence* and *change over time* queries. Dense is ahead on *era/period* queries and, on train, by a wide margin on queries without temporal intent (+0.102, CI excludes 0). Most per-type gaps are not significant on their own.
- **Temporal re-scoring only acts on anchored queries.** It gains +0.022 nDCG@10 (dev) and +0.017 (train) on queries where the extractor finds a time, and is flat on the rest. This follows from its design, since queries without times are left unchanged.
- **Duration and no-temporal-intent queries are the hardest types** for every system (dev hybrid 0.289 and 0.229, against 0.312 for all queries).
- **Significance (dev, macro, Holm-corrected).** Every consecutive step is significant except sparse → dense (+0.022, p = 0.45; the order of the two flips between train and dev). Hybrid beats both of its inputs on nDCG@10 and Recall@100 (p ≤ 0.001). BM25 → BM25+temporal is significant (+0.016, p = 0.02), but only +0.007 without iota.
- **Errors.** Among 25 dev queries where hybrid scores 0, the most common named cause is vocabulary mismatch (8). Next are long, noisy queries (4) and ambiguous queries (2). A wrong time period is the primary cause only once (secondary 3 times). The largest group (9, "other") is mostly near-misses: on-topic non-gold documents, often near-duplicates of each other, fill the top 10 while a gold document sits at ranks 11–50.
- **Near-duplicates are common in the top 10.** 9.7% of hybrid's dev top-10 slots repeat a higher-ranked document (word 5-gram Jaccard ≥ 0.8), in 56% of queries; in history it is 13.6%. v1.1 removed only exact duplicates.

## 1. Query temporal types

**Types.** Seven types, describing what a query asks about time: `specific_date`, `era_period`, `ordering_sequence`, `duration`, `change_over_time`, `present_vs_past`, `no_time` (no temporal intent).

**Train labels** come from `guidance_train.jsonl`, field `temporal_intent` (18 raw values), mapped as follows:

| Type | guidance `temporal_intent` values | Train queries |
| --- | --- | ---: |
| specific_date | when, event_verification_and_authenticity, event_analysis_and_localization | 304 |
| era_period | period_definition, during, historical_attribution_and_context | 250 |
| ordering_sequence | timeline, order, before_after, after, since | 327 |
| duration | duration | 101 |
| change_over_time | origins_evolution_comparative_analysis, trends_changes_and_cross_period | 32 |
| present_vs_past | ongoing_status | 134 |
| no_time | none, why, causation_analysis, or `is_temporal_query` false | 63 |

**Explicit-anchor facet.** Separately, the temporal extractor (`reteco.temporal.extract`, with present markers on) decides whether a query states an explicit time: any extracted expression other than "present". 60% of train and 54% of dev queries are anchored. This facet uses no labels.

**Assigning dev types.** `guidance_dev.jsonl` is never opened; the only guidance path in the code is `guidance_train.jsonl` (`qtypes.load_train_labels`). Dev queries are typed from their `query` text alone. Two methods were compared by stratified 5-fold CV on train:

- hand-written rules: cue patterns on the title, the extractor, then body cues;
- a logistic regression on TF-IDF of the title and of the full cleaned text, the cue patterns as binary features, extractor kind counts, query length and domain one-hot. Classes are balanced, seed 13.

| Method (train 5-fold CV) | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| majority class | 0.270 | 0.061 |
| rules | 0.338 | 0.321 |
| classifier C=0.3 | 0.415 | 0.388 |
| **classifier C=1 (used)** | 0.451 | **0.412** |
| classifier C=3 | 0.459 | 0.405 |
| classifier C=10 | 0.468 | 0.409 |

Per-type F1 of the selected classifier: duration 0.58, specific_date 0.51, era_period 0.49, ordering 0.43, present_vs_past 0.38, change_over_time 0.26, no_time 0.23. The classifier was refit on all of train and applied to dev. Dev counts: era_period 121, specific_date 105, ordering 100, duration 65, no_time 56, present_vs_past 47, change_over_time 25.

**Reading the types.** The guidance labels are noisy: "timeline" covers chronology questions as well as general "how/why" questions, and some "when" queries ask about a mechanism. So 45% accuracy is partly a label-quality ceiling. To keep conclusions honest, every type result is reported twice: on dev with predicted types, and on train with the guidance labels themselves (no prediction noise). Train results with out-of-fold predicted types are in `results/analysis_per_type.csv` too. Types are also confounded with domains: 89% of dev era_period queries are from history, and duration queries are mostly travel, monero, bitcoin and cardano.

Files: `results/analysis_query_types.csv` (one row per train/dev query: type, source, CV prediction, rule type, anchor flag, length, title), `results/analysis_qtype_cv.csv`.

## 2. Per-type and per-domain results (dev)

### By type

Means over the queries of a type (micro); macro over domains would be dominated by cells with 0–3 queries. All numbers are in `results/analysis_per_type.csv`.

nDCG@10:

| Split / type | n | BM25 | +temporal | Sparse | Dense | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev specific_date | 105 | 0.155 | 0.172 | 0.328 | 0.285 | **0.366** |
| dev era_period | 121 | 0.088 | 0.101 | 0.242 | 0.290 | **0.312** |
| dev ordering_sequence | 100 | 0.163 | 0.169 | 0.274 | 0.263 | **0.318** |
| dev duration | 65 | 0.060 | 0.066 | 0.185 | 0.255 | **0.289** |
| dev change_over_time | 25 | 0.124 | 0.136 | 0.310 | 0.264 | **0.339** |
| dev present_vs_past | 47 | 0.114 | 0.133 | 0.247 | 0.245 | **0.296** |
| dev no_time | 56 | 0.055 | 0.050 | 0.188 | 0.195 | **0.229** |
| dev anchored | 280 | 0.119 | 0.141 | 0.280 | 0.290 | **0.338** |
| dev not anchored | 239 | 0.106 | 0.102 | 0.229 | 0.234 | **0.281** |
| train specific_date | 304 | 0.141 | 0.150 | 0.280 | 0.271 | **0.330** |
| train era_period | 250 | 0.060 | 0.071 | 0.232 | 0.247 | **0.289** |
| train ordering_sequence | 327 | 0.094 | 0.111 | 0.252 | 0.210 | **0.280** |
| train duration | 101 | 0.067 | 0.071 | 0.166 | 0.163 | **0.195** |
| train change_over_time | 32 | 0.077 | 0.071 | 0.279 | 0.234 | **0.293** |
| train present_vs_past | 134 | 0.134 | 0.140 | 0.239 | 0.256 | **0.292** |
| train no_time | 63 | 0.076 | 0.078 | 0.132 | **0.234** | 0.213 |

Recall@100 (BM25+temporal equals BM25 by construction):

| Split / type | n | BM25 | Sparse | Dense | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev specific_date | 105 | 0.393 | 0.642 | 0.677 | **0.737** |
| dev era_period | 121 | 0.211 | 0.610 | 0.668 | **0.701** |
| dev ordering_sequence | 100 | 0.327 | 0.575 | 0.584 | **0.650** |
| dev duration | 65 | 0.368 | 0.699 | 0.683 | **0.760** |
| dev change_over_time | 25 | 0.332 | 0.678 | 0.685 | **0.745** |
| dev present_vs_past | 47 | 0.425 | 0.648 | 0.663 | **0.705** |
| dev no_time | 56 | 0.255 | 0.621 | 0.525 | **0.686** |

**Wins and losses by system** (paired differences with 95% query-bootstrap CIs in `results/analysis_type_deltas.csv`; "sig" = CI excludes 0):

| Comparison | Wins (both splits, same sign) | Loses | Inconsistent |
| --- | --- | --- | --- |
| BM25+temporal vs BM25 | anchored (+0.022 dev sig, +0.017 train sig), era_period (sig both), specific_date, ordering (train sig), present_vs_past (dev sig) | none | change_over_time, no_time (−0.005 dev, +0.002 train) |
| Dense vs Sparse | era_period (+0.048 dev sig, +0.016 train), no_time (+0.007 dev, +0.102 train sig) | specific_date (−0.042 / −0.009), ordering (−0.011 / −0.042 train sig), change_over_time (−0.047 / −0.045) | duration (+0.070 dev sig, −0.002 train), present_vs_past |
| Hybrid vs Sparse | every type, both splits (sig for specific_date, era_period, ordering on both) | none | none |
| Hybrid vs Dense | every type on dev (sig for specific_date, ordering, change, present) | no_time on train (−0.022, n.s.) | |

Interpretation: types whose answers hinge on named entities, exact dates or the order of named events favour exact term matching. Period-style and intent-free questions favour the dense model's paraphrase matching. The two retrievers fail on different queries, which is why the fusion wins everywhere. BM25 is weakest on era_period queries (dev 0.088, R@100 0.21): BM25 matches the many incidental years and names in long history posts. Tuning and query cleaning give their largest relative gain there.

### By domain

`results/analysis_per_domain.csv`, figure `paper/figures/per_domain.pdf`, paper Table `tab:per-domain`.

| Domain | n | BM25 | +temp. | Sparse | Dense | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bitcoin | 30 | 0.038 | 0.038 | 0.176 | 0.169 | **0.215** |
| cardano | 15 | 0.103 | 0.100 | 0.237 | 0.194 | **0.258** |
| economics | 25 | 0.047 | 0.057 | 0.169 | 0.153 | **0.195** |
| genealogy | 35 | 0.195 | 0.225 | 0.348 | 0.319 | **0.370** |
| history | 240 | 0.087 | 0.101 | 0.246 | 0.270 | **0.293** |
| hsm | 45 | 0.249 | 0.238 | 0.388 | 0.311 | **0.455** |
| iota | 3 | 0.331 | 0.447 | 0.518 | **0.713** | 0.658 |
| law | 11 | 0.057 | 0.069 | 0.220 | 0.344 | **0.345** |
| monero | 19 | 0.032 | 0.055 | 0.132 | 0.091 | **0.174** |
| politics | 45 | 0.263 | 0.259 | 0.362 | 0.340 | **0.471** |
| quant | 10 | 0.024 | 0.031 | **0.104** | 0.095 | 0.100 |
| travel | 30 | 0.044 | 0.053 | 0.163 | 0.250 | **0.256** |
| workplace | 11 | 0.023 | 0.024 | 0.240 | 0.346 | **0.355** |

nDCG@10 only; per-domain Recall@100 for every system is in the CSV and in the paper table. Hybrid is best, or within 0.004 of the best, in 12 of 13 domains. The exception is iota (3 queries), where dense alone is better.

## 3. Error analysis

**Sample.** 176 of 519 dev queries have hybrid nDCG@10 = 0 (the pool is "< 0.05", which here equals "= 0"). I drew 25, at least one from each of the 12 domains with failures and the rest proportional to each domain's share of the pool (largest remainder), with seed 13: history 7, every other domain 1–2 (`errors.stratified_sample`). For each query, `scripts/analysis_errors.py` collects automatic signals: gold ranks in the hybrid top 100 and in the depth-1000 sparse/dense candidate lists, gold length, query-term coverage, boilerplate cue hits, and the temporal fit of gold vs. top-3 documents. It also writes a dossier with the query, gold documents and hybrid's top 3. I read every dossier and assigned one primary and at most one secondary category. Labels and notes: `results/analysis_error_labels.csv`; signals: `results/analysis_error_candidates.csv`.

| Category | Primary | Secondary | Example |
| --- | ---: | ---: | --- |
| Vocabulary mismatch | 8 | 0 | politics `83771_988`: "officials who served two countries in quick succession (Fischer)"; the gold documents are about Carney and Jaresko, whom the query never names. |
| Very long noisy query | 4 | 2 | history `70471_283`: 347 words, mostly a Casanova anecdote, so Casanova biographies win; the gold covers laws against usurping noble titles. |
| Wrong time period | 1 | 3 | history `72255_172`: the query's dates (1603, 1707, 1801) are counter-examples, so Acts-of-Union pages win; the gold covers 1859 Moldavia–Wallachia and 1926–32 Hejaz–Nejd (temporal fit 0.39 for gold vs. 0.78 for the top 3). |
| Gold doc is boilerplate-like | 1 | 2 | workplace `200148_1194`: the only gold is law-firm marketing copy ("our team is always at..."); forum answers rank above it. |
| Ambiguous query | 2 | 0 | hsm `5514_757`: "when did mathematicians start drawing figures from equations?" Top docs give Descartes/Fermat 1637; the gold points to Newton's cubics and Euler. |
| Other | 9 | 0 | history `76410_105`: ranks 1–3 discuss the economic *effects* of Protestant iconoclasm (2 and 3 are near-identical); the gold, on motives, is at 25. |

"Other" breaks down into 7 near-misses (on-topic non-gold documents in the top 10, gold at 11–50), 1 fusion loss (bitcoin `124510_9`: sparse had a gold passage at rank 3, dense never retrieved it, and RRF moved it to 19), and 1 very long gold document (hsm `5709_750`: 5,512 words, only the first 512 tokens are encoded).

Observations:

- **Most failures are near-misses, not misses.** In all 25 queries some gold document is in the depth-1000 candidates of sparse or dense, and in 19 it is in hybrid's top 100 (median best rank 25, range 11–53). A reranker that reads the full query and passage should be able to recover several of them. Long noisy queries (truncated at 512 tokens) and boilerplate-like gold will remain hard.
- **Temporal mismatch is rarely the main cause** of zero-score failures. When dates do mislead, it is because the query mentions times it is *not* asking about (counter-examples, log timestamps, scenario years such as "a cross-section in 2016 and 2018"). Temporal re-scoring would push the wrong way in those cases. This fits its small measured gains.
- **Type errors are visible:** travel `185727_1169` ("translate before or after the apostille?") was typed `duration` but is an ordering question.

### Near-duplicates in top-10 lists

`scripts/analysis_neardup.py` (no qrels) counts top-10 slots whose document repeats a higher-ranked one: *strict* means identical after lowercasing and dropping non-alphanumerics; *shingle* means word 5-gram Jaccard ≥ 0.8.

| System (dev) | Strict | Shingle | Queries with ≥ 1 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7% | 4.4% | 26% |
| BM25+temporal | 0.9% | 4.4% | 28% |
| Sparse (tuned) | 4.8% | 9.6% | 52% |
| Dense | 7.1% | 11.3% | 61% |
| Hybrid | 5.7% | 9.7% | 56% |

Better systems retrieve more near-duplicates, because the duplicated texts are on-topic. In hybrid's top 10, history has 13.6%, bitcoin 12.7% and travel 11.3%. Collapsing near-duplicates would free top-10 slots. This is untested, would be a system change, and is left for the system agents.

## 4. Significance tests

`scripts/analysis_significance.py`, `results/analysis_significance.csv`. Both tests use the macro statistic: the difference of two systems is Σ w_i d_i with w_i = 1 / (13 · n_domain(i)).

- **Paired randomization test:** 10,000 sign flips of the per-query differences, two-sided, p = (hits + 1) / (10,001).
- **Paired bootstrap:** 10,000 resamples of queries within each domain (domain sizes fixed), 95% percentile CI.
- **Holm correction** over the seven pairs of each metric.
- Every test is repeated without iota.

nDCG@10, dev:

| Pair | Δ macro | 95% CI | p (Holm) | Δ without iota | p without iota |
| --- | ---: | --- | ---: | ---: | ---: |
| bm25 → bm25+temporal | +0.0157 | [+0.0051, +0.0266] | 0.020 | +0.0073 | 0.040 |
| bm25+temporal → sparse_best | +0.1236 | [+0.0982, +0.1492] | < 0.001 | +0.1280 | < 0.001 |
| sparse_best → dense_best | +0.0224 | [−0.0209, +0.0718] | 0.446 | +0.0080 | 0.622 |
| dense_best → hybrid_best | +0.0424 | [+0.0198, +0.0644] | 0.001 | +0.0506 | < 0.001 |
| bm25 → sparse_best * | +0.1393 | [+0.1111, +0.1673] | < 0.001 | +0.1353 | < 0.001 |
| bm25 → dense_best * | +0.1617 | [+0.1257, +0.2003] | < 0.001 | +0.1433 | < 0.001 |
| sparse_best → hybrid_best * | +0.0648 | [+0.0344, +0.0993] | < 0.001 | +0.0586 | < 0.001 |

Recall@100, dev: sparse → dense −0.027 (p = 0.59); dense → hybrid +0.090, sparse → hybrid +0.063, bm25 → sparse +0.249 (all p < 0.001). BM25 → temporal is 0 by construction.

\* not consecutive in the progression. The 10k permutations bound p below at about 10⁻⁴; "< 0.001" means no permuted statistic reached the observed one, or only a handful did.

**Reading.** Sparse vs. dense is a tie: the dev ordering (dense ahead) is the opposite of train (sparse ahead, 0.2435 vs. 0.2261), and the CI spans ±0.05. The temporal gain is real but small, and half of it is iota.

## 5. Figures

`scripts/make_figures.py` writes PDF and PNG files to `paper/figures/`. It reads only `results/*.csv`.

- `per_domain`: dev nDCG@10 per domain (with dev query counts) and the macro, for all five systems.
- `per_type`: nDCG@10 per type; top panel dev with predicted types, bottom panel train with guidance types.
- `progression`: macro nDCG@10 and Recall@100 along bm25 → +temporal → sparse → dense → hybrid, train and dev, in two panels (no dual axis).

Colours are slots 1–5 of a categorical palette checked for colour-vision-deficiency separation (worst adjacent ΔE 9.1). Three of the five fall below 3:1 contrast on white, so every plotted value is also in a paper table.

## 6. Paper skeleton

`paper/main.tex` (ACL style; `acl.sty` and `acl_natbib.bst` copied unchanged from acl-org/acl-style-files), `paper/references.bib`, `paper/README.md`.

- Sections: abstract placeholder, introduction, task, data, system (BM25 tuning, query cleaning, dense, fusion, temporal re-scoring, reranker placeholder), results, analysis, limitations, conclusion.
- The limitations section includes the temporal-extractor bug fixed after viewing dev, iota's 3 dev queries and the other small dev domains, approximate types, single-annotator error analysis, and repeated dev reporting.
- **Every table is generated** by `paper/make_tables.py` from `results/*.csv` into `paper/tables/*.tex`. Numbers quoted in the prose are macros in `paper/tables/numbers.tex`, generated the same way.
- **Reranker hooks:** red `\TODO` markers in the abstract, introduction, §4.6, results, figure captions, type analysis, error analysis, limitations and conclusion. `make_tables.py` adds any 13-domain `experiments.csv` row named `rerank*` to the main table automatically; until then the table shows a TODO row.
- Not compiled: there is no LaTeX installation on this machine. I checked brace and environment balance, that every `\ref` has a `\label`, that every `\cite` key exists in the .bib, and that the figure files exist.

## Reproduce

```bash
export RETECO_DATA=data/data/track1_tempo PYTHONPATH="$PWD/src"
python scripts/analysis_query_types.py      # ~40 s; types + CV
python scripts/analysis_breakdown.py        # per-query / per-type / per-domain / deltas
python scripts/analysis_significance.py
python scripts/analysis_errors.py --dossier /tmp/error_dossier.md   # sample + signals
python scripts/analysis_neardup.py          # ~1 min, streams the corpora
python scripts/analysis_data_stats.py
/opt/homebrew/bin/python3 scripts/make_figures.py   # any Python with matplotlib
python paper/make_tables.py
python -m pytest tests/test_analysis_tools.py
```

## Decisions made

1. **Type definitions from `temporal_intent`.** The seven requested types are defined by mapping the guidance field `temporal_intent` (table in §1). The long tail of rare intents goes to the nearest type, e.g. event verification ("did X really happen in year Y") → specific_date. "No explicit time" became **no temporal intent** (`none`, `why`, `causation_analysis`, or `is_temporal_query` false). The literal surface reading ("the query states no time") is reported as a separate, label-free facet (explicit anchor yes/no, from the extractor), because it is orthogonal to the other six types: a "when did X happen" query usually has no explicit time.
2. **"Cluster" means group by type.** Train queries are grouped by guidance-derived types, not clustered unsupervised. Unsupervised clusters would not line up with the requested types.
3. **Classifier over rules** for dev assignment, chosen by train-CV macro-F1 (0.412 vs. 0.321). C = 1 is chosen inside the same CV, so the CV figure is slightly optimistic.
4. **Class-balanced weights kept.** An unweighted model is about 2 points more accurate (0.457–0.477) but predicts change_over_time for only 4–6 of 1,211 train queries, which would leave that dev group nearly empty. Balanced weights keep all seven groups populated, at the cost of lower precision for the rare classes (no_time precision 0.18).
5. **Per-type scores are query means (micro), not domain macros**, since many type × domain cells hold 0–3 queries. Per-domain scores and the macro row follow the official definition.
6. **Every type result is shown on train with guidance labels too**, because dev types are only ~45% accurate. A finding is called consistent only if it has the same sign on both.
7. **Progression order** is bm25 → bm25+temporal → sparse_best → dense_best → hybrid_best. This follows the order of construction and the paper's narrative (temporal re-scoring is applied to BM25), and dev macro scores happen to increase along it. The pair "bm25+temporal → sparse_best" is consecutive in the chart but not a real build step, so three more pairs (bm25 → sparse, bm25 → dense, sparse → hybrid) are also tested. Holm correction covers all seven pairs per metric.
8. **Both tests on the official macro statistic**, with domain-stratified bootstrap. Query-level (micro) differences are only used inside the per-type tables.
9. **Error-analysis pool = nDCG@10 < 0.05**, which on dev equals the 176 zero-score queries. The 25 were stratified by domain with seed 13, not picked by hand, so the categories describe typical failures rather than striking ones.
10. **"Other" was split** into near-miss, fusion loss and long gold document in a separate column, because near-misses are the largest group and the requested categories have no slot for them. The categories, and which example represents each, are my judgment as a single annotator. The dossiers quote dev gold text, so they are written outside the repo (`--dossier` is required) and are not committed.
11. **Dev gold documents were read** for the error analysis, as the task requires. No run, config or parameter was changed afterwards. The near-duplicate finding is reported as a suggestion only.
12. **Near-duplicate measurement** uses two thresholds (strict and 5-gram Jaccard ≥ 0.8) on top-10 lists. The threshold was not tuned; the qualitative result (10% of slots, rising with system quality) holds under both.
13. **Data path.** All runs use `RETECO_DATA=data/data/track1_tempo` as instructed. The depth-1000 candidate lists for the error signals are read from `cache/dense/candidates/` (read-only; nothing in `cache/` was written).
14. **Figures run outside the project venv.** The venv has no matplotlib, and I did not install packages into a venv shared with the running GPU job. `make_figures.py` needs only the standard library and matplotlib and was run with Homebrew's Python 3.14 (matplotlib 3.10.8).
15. **No rows appended to `results/experiments.csv`.** No new run was produced or scored, and analysis results do not fit its schema (one row per run). They live in `results/analysis_*.csv` and `results/data_stats.csv`.
16. **Paper data table from a new CSV.** `scripts/analysis_data_stats.py` writes `results/data_stats.csv` so the data table is generated like every other table. Its numbers match `docs/DATA.md`.
17. **ACL style files vendored** into `paper/`, unchanged, so the draft builds without fetching anything. The bibliography only contains entries I am confident of; the RETECO task paper, TEMPO and the reranker are left as TODOs rather than guessed.
18. **New test file** `tests/test_analysis_tools.py` (10 tests: macro weighting, both tests, Holm, sampler, rank order, type mapping and rules, per-query scorer on the tiny fixture). No existing file was modified.

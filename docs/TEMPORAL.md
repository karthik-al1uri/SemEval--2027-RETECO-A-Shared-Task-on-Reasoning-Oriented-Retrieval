# Temporal-fit reranking

Goal: rerank a candidate list so that documents on the right topic but the wrong time drop. Code is in `src/reteco/temporal/`, configs are `configs/temporal_*.yaml`, runs are `runs/<split>/bm25+temporal_*.trec`, and scores are in `results/experiments.csv`.

## Summary

- **Best system on train: `temporal_fit_present`.** BM25 macro nDCG@10 goes from **0.1076 to 0.1171 on train** and from **0.1148 to 0.1304 on dev** (+0.0157, paired bootstrap 95% CI [+0.0055, +0.0267]). All parameters were tuned on train; dev was scored once per variant.
- Part of the dev macro gain comes from iota, which has only 3 dev queries. Without iota the dev gain is **+0.0073**, and the query-weighted (micro) gain is **+0.0101**. On both splits, about 3 times as many queries improve as get worse (dev: 60 better, 19 worse, 440 unchanged).
- Gains are consistent across both splits in **history (+0.014 dev), genealogy (+0.030), economics (+0.010) and law (+0.011)**. **Politics does not gain** (−0.004 dev). The train analysis shows why: in politics, gold documents fit the query's time no better than BM25's other candidates (AUC 0.52).
- The useful signal is **concentration, not overlap**. BM25 already matches year tokens, so most non-gold candidates mention the query's year somewhere (in history, 84% overlap for non-gold vs 78% for gold). What separates gold from non-gold is the share of a document's time mentions that sit near the query's time.
- The best tuned setting is **penalty-only** (`center = 1`): wrong-time documents drop, time-matching documents and documents without times stay where they are. Boosting only (`center = 0`) is weaker on train and below baseline on dev.
- The **guidance-trained gate classifier adds nothing**. 97% of queries in which the extractor finds a time are classified as calendar-anchored anyway, so the gate mostly repeats "a time expression was found".
- The rescorer reads **any TREC run** (`--run`), so it can be stacked on dense or hybrid runs.

## 1. Time extraction (`extract.py`)

The extractor uses rules (regex), with dateparser for explicit date strings. Each expression becomes a closed year interval `[start, end]`; negative years are BCE. It handles these kinds, in priority order (a matched span is masked so later patterns cannot match inside it):

| Kind | Examples | Interval |
| --- | --- | --- |
| date | "5 March 1943", "March 1943", "1943-03-05", "12/05/1943" | that year (spans normalised by `dateparser.parse`, memoised) |
| range | "1914-1918", "1914–18", "from 1914 to 1918", "between 2000 and 2010", "221–206 BC" | [start, end] |
| century / millennium | "19th century", "early nineteenth-century", "3rd century BC", "1800s", "2nd millennium BC" | early/mid/late = first/middle/last third; "first half of" etc. |
| decade | "1940s", "the late 1950s", "the '60s", "the sixties" | [1950, 1959], late → [1957, 1959] |
| year | standalone 1000–2029; any year with an uppercase era marker ("44 BC", "AD 79", "476 AD", "10,000 BCE") | [y, y] |
| era | gazetteer of about 170 named periods and events: wars (WWII, Cold War, Thirty Years' War), periods (Victorian, Middle Ages, Renaissance, Bronze Age, Ming dynasty, Edo period), events (Great Depression, 9/11, Brexit, COVID), crypto milestones (SegWit, Taproot, Cardano Shelley/Alonzo, RingCT, IOTA Chrysalis) | conventional dates |
| relative | "before / prior to / pre- / until X", "after / since / post- X", where X is any of the kinds above | before X → [−12000, X.start]; after X → [X.end, 2026] |
| present (optional, queries only) | "nowadays", "currently", "today", "these days", "still used" | [2022, 2026] |

Plain 4-digit numbers are filtered by context, because bitcoin, monero and travel are full of quantities in the year range. A number is not treated as a year when:
- it follows a currency sign, `#`, "block", "height", "port", "version", "BIP", "page" and similar prefixes;
- it is followed by a unit ("BTC", "blocks", "sats", "people", "dollars", "%", ...);
- it is part of a longer number, decimal or identifier.

Era markers on bare years must be uppercase and must not run into a following letter or digit. Extractor v1 parsed "25 ADA", "AD1" (aggregate demand in economics posts) and hex fragments such as "8C E" as early years. Section 6 explains how this was found.

**Speed.** Pure `re`, about 1.3 ms per document on bitcoin and 3.5 ms per document on history (long documents), single core. The main cost was the 170-pattern gazetteer. Each era pattern now runs only if one of its required literals (derived automatically, e.g. "victorian" or "ww") appears in the lowercased text. I checked on 2,000 history documents that this skips no match. `dateparser.search.search_dates` took 6.3 ms per history document even when truncated to 3,000 characters, and it returns only point dates (no ranges, centuries, eras or relative expressions). So dateparser only normalises the short date spans the regexes find.

**Caching (`doctimes.py`).** Document times are cached in `cache/temporal/doc_times/v2/<domain>.jsonl`, one line per document id, and filled on demand. Rescoring a new run extracts only documents not seen before. The BM25 train and dev top-100 lists contain 173k (query, document) pairs but only 48k unique documents; extracting all 13 domains took about 6 minutes on 8 workers on a loaded machine.

## 2. Analysis on TRAIN (`analyze.py`)

Gold labels are used here only to describe the data. "Time" means at least one extracted expression, excluding "present". The overlap, distance and similarity columns use only queries with a time and documents with a time. "Within τ" means the closest gap is at most 10 years. "Mean sim" is the average of doc_prox and q_cover (τ = 10, defined in section 3). AUC is the per-query probability that a gold document in the BM25 top 100 scores above a non-gold one (documents without times score 0), averaged over queries.

| Domain | Queries | Q w/ time | Gold w/ time | Non-gold top-100 w/ time | Overlap gold / non-gold | Within τ gold / non-gold | Mean sim gold / non-gold | AUC (top-100) | AUC queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bitcoin | 70 | 21.4% | 27.0% | 74.6% | 66.7% / 74.5% | 100.0% / 96.9% | 0.877 / 0.786 | 0.576 | 7 |
| cardano | 36 | 5.6% | 28.7% | 47.0% | – / 59.4% | – / 93.4% | – / 0.749 | 0.242 | 1 |
| economics | 58 | 55.2% | 67.3% | 90.3% | 73.5% / 75.8% | 88.0% / 88.4% | 0.753 / 0.665 | 0.713 | 17 |
| genealogy | 80 | 93.8% | 87.0% | 95.8% | 80.9% / 82.8% | 88.1% / 88.0% | 0.688 / 0.611 | 0.628 | 42 |
| history | 561 | 73.1% | 77.6% | 96.3% | 78.2% / 84.0% | 85.3% / 87.3% | 0.725 / 0.602 | 0.681 | 140 |
| hsm | 105 | 63.8% | 92.5% | 89.3% | 71.6% / 73.4% | 85.2% / 80.4% | 0.593 / 0.521 | 0.646 | 51 |
| iota | 7 | 0.0% | 34.5% | 79.0% | – / – | – / – | – / – | – | 0 |
| law | 24 | 41.7% | 50.0% | 82.1% | 92.9% / 66.7% | 92.9% / 76.1% | 0.742 / 0.578 | 0.569 | 6 |
| monero | 46 | 34.8% | 30.6% | 69.2% | 52.4% / 73.3% | 100.0% / 97.3% | 0.901 / 0.819 | 0.545 | 5 |
| politics | 105 | 66.7% | 87.6% | 94.8% | 78.7% / 86.9% | 89.0% / 94.6% | 0.757 / 0.775 | 0.522 | 56 |
| quant | 24 | 25.0% | 11.9% | 70.2% | 66.7% / 48.1% | 66.7% / 75.3% | 0.569 / 0.512 | 0.093 | 2 |
| travel | 70 | 28.6% | 16.9% | 65.7% | 66.7% / 64.4% | 100.0% / 89.5% | 0.870 / 0.756 | 0.528 | 7 |
| workplace | 25 | 16.0% | 24.7% | 65.1% | 33.3% / 44.6% | 66.7% / 77.7% | 0.538 / 0.565 | 0.131 | 1 |

Expression kinds on train: queries have 939 years, 600 eras, 355 dates, 175 centuries, 168 relative, 120 ranges and 85 decades. Gold documents have 9,812 years, 3,391 dates, 2,987 eras, 1,990 ranges, 1,158 centuries, 946 relative and 692 decades.

What this says about the signal:

1. **Coverage.** 60% of train queries (727/1211) have an extracted time, against 43.6% with an explicit year in `docs/DATA.md`; eras, centuries and decades add the rest. Genealogy (94%), history (73%), politics (67%), hsm (64%) and economics (55%) are the domains where the score can act at all. In iota, cardano and workplace, almost no query has a time, so the score rarely fires there.
2. **BM25 candidates are time-rich.** Non-gold top-100 documents carry time expressions more often than gold documents (history: 96% vs 78%; bitcoin: 75% vs 27%). Long, date-heavy queries pull in date-heavy documents. So "has a time" is not a positive signal, and treating documents without times as neutral (rather than worse) matters.
3. **Overlap is not discriminative; concentration is.** Because BM25 already matches the year tokens, most candidates mention the query's year somewhere. The binary "any overlap" rate is even *lower* for gold than for non-gold in history, politics, monero and genealogy. The graded similarity, which asks how much of a document's time is near the query's, is higher for gold in every large domain except politics, with within-query AUC 0.68 (history), 0.71 (economics), 0.65 (hsm) and 0.63 (genealogy).
4. **Politics is the exception** (AUC 0.52, mean sim gold 0.757 vs non-gold 0.775). Politics queries mention years, but the gold answers are often general or current explanations whose own dates do not concentrate around the query's. Expect little gain there, which the results confirm.

## 3. Temporal-fit score and rescoring (`fit.py`, `rescore.py`)

For query intervals Q_i and document intervals D_j:

```
dist(i, j) = max(0, D_j.start − Q_i.end, Q_i.start − D_j.end)      # 0 when they overlap
k(i, j)    = exp(−dist(i, j) / τ)
doc_prox   = mean_j max_i k(i, j)    # share of the doc's time mentions near the query time
q_cover    = mean_i max_j k(i, j)    # share of the query's anchors the doc covers
best       = max_ij k(i, j)
sim        = (w_doc·doc_prox + w_q·q_cover + w_best·best) / (w_doc + w_q + w_best)
temporal   = sim − center   if both query and doc have times, else 0   (neutral)
final      = a · norm(retrieval) + b · g(q) · temporal
```

- `norm` is a per-query min-max or z-score of the input run's scores, so BM25, dense and hybrid runs share one scale. `a = 1`; only `b` is tuned.
- `g(q)` is an optional gate (section 5); without a gate it is 1.
- `max_width` drops intervals wider than this many years (open-ended "before 1600" or "Stone Age" intervals) from the comparison.
- "No time information" is always exactly 0 and is never penalised: a document without times keeps its rank relative to other documents with the same retrieval score. With `center = 1`, matching documents also score 0 and only wrong-time documents go negative.

Tuning (`tune.py`) is a full grid on train macro nDCG@10 over τ ∈ {3, 10, 25, 50, 100}, 7 weight mixes, center ∈ {0, 0.25, 0.5, 0.75, 1}, norm ∈ {minmax, zscore}, max_width ∈ {none, 500} and b ∈ {0.02 … 12}: 9,800 settings per variant, about 90 s each. It uses a vectorised nDCG@10 that matches `pytrec_eval` exactly (checked on the small domains and in `tests/test_temporal.py`). The b curves are smooth with a broad plateau; for `fit`, train macro stays at or above 0.113 for every b from 0.75 to 3, and peaks at 0.1155 at b = 1.5.

Selected settings (all from train):

| Variant | What it tests | τ | weights (doc, q, best) | center | norm | b |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `boost` | center fixed at 0: only boosts | 10 | (1, 0, 0) | 0 | minmax | 0.75 |
| `fit` | center tuned | 100 | (1, 0, 0) | 1 | minmax | 1.5 |
| `fit_years` | ablation: no eras, relative or present (explicit dates, years, ranges, decades and centuries only) | 10 | (1, 0, 0) | 0.5 | zscore | 5 |
| `fit_present` | `fit` + query "nowadays/currently/today" → [2022, 2026] | 25 | (1, 0, 0) | 1 | zscore | 5 |
| `fit_gated` | `fit` with b × classifier probability | 100 | (1, 0, 0) | 1 | minmax | 1.5 |
| `fit_gated_hard` | `fit` only for queries the classifier marks calendar-anchored | 100 | (1, 0, 0) | 1 | minmax | 1.5 |

Every variant picks `doc_prox` alone. Query coverage and the best single match add nothing: BM25 already secures that the query's year appears somewhere. The useful question is whether the document is mostly about that time.

## 4. Results

nDCG@10, BM25 top 100 rescored (Δ vs BM25 in brackets; blank Δ = unchanged).

#### train

| System | Macro | bitcoin | cardano | economics | genealogy | history | hsm | iota | law | monero | politics | quant | travel | workplace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.1076 | 0.0856 | 0.1848 | 0.0382 | 0.1243 | 0.0659 | 0.1735 | 0.1563 | 0.0946 | 0.0382 | 0.2810 | 0.0255 | 0.0504 | 0.0798 |
| bm25+temporal_boost | 0.1130 (+0.0054) | 0.0856 | 0.1766 (-0.008) | 0.0470 (+0.009) | 0.1500 (+0.026) | 0.0771 (+0.011) | 0.1930 (+0.020) | 0.1563 | 0.1203 (+0.026) | 0.0387 (+0.001) | 0.2679 (-0.013) | 0.0255 | 0.0515 (+0.001) | 0.0798 |
| bm25+temporal_fit | 0.1155 (+0.0079) | 0.0856 | 0.1848 | 0.0516 (+0.013) | 0.1532 (+0.029) | 0.0793 (+0.013) | 0.1789 (+0.005) | 0.1563 | 0.1275 (+0.033) | 0.0387 (+0.001) | 0.2812 | 0.0302 (+0.005) | 0.0536 (+0.003) | 0.0798 |
| bm25+temporal_fit_years | 0.1144 (+0.0068) | 0.0856 | 0.1812 (-0.004) | 0.0612 (+0.023) | 0.1471 (+0.023) | 0.0807 (+0.015) | 0.1897 (+0.016) | 0.1563 | 0.1253 (+0.031) | 0.0381 | 0.2663 (-0.015) | 0.0255 | 0.0508 | 0.0798 |
| **bm25+temporal_fit_present** | **0.1171 (+0.0095)** | 0.0856 | 0.1854 (+0.001) | 0.0491 (+0.011) | 0.1502 (+0.026) | 0.0796 (+0.014) | 0.1778 (+0.004) | 0.1970 (+0.041) | 0.1134 (+0.019) | 0.0387 (+0.001) | 0.2815 | 0.0255 | 0.0538 (+0.003) | 0.0851 (+0.005) |
| bm25+temporal_fit_gated | 0.1155 (+0.0079) | 0.0856 | 0.1848 | 0.0502 (+0.012) | 0.1532 (+0.029) | 0.0792 (+0.013) | 0.1790 (+0.006) | 0.1563 | 0.1275 (+0.033) | 0.0387 (+0.001) | 0.2829 (+0.002) | 0.0302 (+0.005) | 0.0536 (+0.003) | 0.0798 |
| bm25+temporal_fit_gated_hard | 0.1155 (+0.0079) | 0.0856 | 0.1848 | 0.0516 (+0.013) | 0.1532 (+0.029) | 0.0793 (+0.013) | 0.1789 (+0.005) | 0.1563 | 0.1275 (+0.033) | 0.0387 (+0.001) | 0.2812 | 0.0302 (+0.005) | 0.0536 (+0.003) | 0.0798 |

#### dev

| System | Macro | bitcoin | cardano | economics | genealogy | history | hsm | iota | law | monero | politics | quant | travel | workplace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.1148 | 0.0384 | 0.1027 | 0.0467 | 0.1949 | 0.0870 | 0.2489 | 0.3309 | 0.0574 | 0.0320 | 0.2625 | 0.0237 | 0.0442 | 0.0230 |
| bm25+temporal_boost | 0.1118 (-0.0030) | 0.0409 (+0.003) | 0.1027 | 0.0555 (+0.009) | 0.2078 (+0.013) | 0.1050 (+0.018) | 0.2476 (-0.001) | 0.2083 (-0.123) | 0.0700 (+0.013) | 0.0578 (+0.026) | 0.2590 (-0.004) | 0.0307 (+0.007) | 0.0428 (-0.001) | 0.0252 (+0.002) |
| bm25+temporal_fit | 0.1294 (+0.0146) | 0.0384 | 0.1027 | 0.0565 (+0.010) | 0.2429 (+0.048) | 0.1026 (+0.016) | 0.2329 (-0.016) | 0.4360 (+0.105) | 0.0686 (+0.011) | 0.0481 (+0.016) | 0.2483 (-0.014) | 0.0307 (+0.007) | 0.0515 (+0.007) | 0.0230 |
| bm25+temporal_fit_years | 0.1273 (+0.0125) | 0.0384 | 0.1027 | 0.0562 (+0.010) | 0.2398 (+0.045) | 0.1035 (+0.017) | 0.2470 (-0.002) | 0.3934 (+0.062) | 0.0807 (+0.023) | 0.0320 | 0.2713 (+0.009) | 0.0307 (+0.007) | 0.0360 (-0.008) | 0.0230 |
| **bm25+temporal_fit_present** | **0.1304 (+0.0156)** | 0.0384 | 0.0996 (-0.003) | 0.0568 (+0.010) | 0.2247 (+0.030) | 0.1011 (+0.014) | 0.2383 (-0.011) | 0.4469 (+0.116) | 0.0686 (+0.011) | 0.0549 (+0.023) | 0.2587 (-0.004) | 0.0307 (+0.007) | 0.0533 (+0.009) | 0.0236 (+0.001) |
| bm25+temporal_fit_gated | 0.1281 (+0.0133) | 0.0384 | 0.1027 | 0.0565 (+0.010) | 0.2429 (+0.048) | 0.1040 (+0.017) | 0.2333 (-0.016) | 0.4292 (+0.098) | 0.0686 (+0.011) | 0.0362 (+0.004) | 0.2483 (-0.014) | 0.0307 (+0.007) | 0.0515 (+0.007) | 0.0230 |
| bm25+temporal_fit_gated_hard | 0.1294 (+0.0146) | 0.0384 | 0.1027 | 0.0565 (+0.010) | 0.2429 (+0.048) | 0.1026 (+0.016) | 0.2329 (-0.016) | 0.4360 (+0.105) | 0.0686 (+0.011) | 0.0481 (+0.016) | 0.2483 (-0.014) | 0.0307 (+0.007) | 0.0515 (+0.007) | 0.0230 |

Robustness. "Macro Δ" is relative to BM25. The CI comes from a paired bootstrap that resamples queries within each domain (2,000 samples, seed 13). "Micro Δ" is the mean over all queries. Recall@100 is unchanged by construction (0.3365 train, 0.4045 dev).

| Split | System | Macro Δ | 95% CI | Macro Δ w/o iota | Micro Δ | Queries better / worse / same |
| --- | --- | ---: | --- | ---: | ---: | --- |
| train | boost | +0.0055 | [+0.0010, +0.0108] | +0.0059 | +0.0082 | 118 / 56 / 1037 |
| train | fit | +0.0079 | [+0.0038, +0.0121] | +0.0086 | +0.0102 | 121 / 35 / 1055 |
| train | fit_years | +0.0069 | [+0.0022, +0.0119] | +0.0075 | +0.0101 | 102 / 48 / 1061 |
| train | fit_present | +0.0096 | [+0.0038, +0.0155] | +0.0070 | +0.0100 | 125 / 42 / 1044 |
| train | fit_gated | +0.0079 | [+0.0041, +0.0124] | +0.0086 | +0.0103 | 122 / 34 / 1055 |
| train | fit_gated_hard | +0.0079 | [+0.0042, +0.0120] | +0.0086 | +0.0102 | 121 / 35 / 1055 |
| dev | boost | -0.0030 | [-0.0127, +0.0075] | +0.0070 | +0.0100 | 51 / 17 / 451 |
| dev | fit | +0.0146 | [+0.0051, +0.0248] | +0.0071 | +0.0103 | 58 / 17 / 444 |
| dev | fit_years | +0.0125 | [+0.0032, +0.0231] | +0.0083 | +0.0122 | 47 / 21 / 451 |
| dev | fit_present | +0.0157 | [+0.0055, +0.0267] | +0.0073 | +0.0101 | 60 / 19 / 440 |
| dev | fit_gated | +0.0133 | [+0.0040, +0.0216] | +0.0062 | +0.0105 | 57 / 17 / 445 |
| dev | fit_gated_hard | +0.0146 | [+0.0052, +0.0241] | +0.0071 | +0.0103 | 58 / 17 / 444 |

Observations:

- **By domain.** History (+0.013 to +0.017), genealogy (+0.023 to +0.048), economics (+0.009 to +0.023) and law (+0.011 to +0.033) gain under every penalty variant on both splits. Hsm gains on train but loses 0.002 to 0.016 on dev. Politics is flat or slightly negative, as the analysis predicted (AUC 0.52). Bitcoin, quant, travel and workplace barely move, because few of their queries carry a time.
- **Iota dominates the dev macro.** It has 3 dev queries, so a single query moving from 0.46 to 0.70 adds about +0.006 to the macro average (0.24 / 3 / 13). Without iota, all penalty variants give about +0.006 to +0.008 on dev, close to train. Read the macro deltas with that in mind.
- **Penalise, don't boost.** `boost` (center 0) helps on train but is negative on dev macro: it loses 0.123 on dev iota, and on train it already lost 0.013 in politics and 0.008 in cardano. Boosting moves documents up for merely mentioning the right year, which BM25 already rewards. Penalty-only (`center = 1`) moves only clearly wrong-time documents down.
- **Eras and relative expressions: mixed.** `fit` vs `fit_years` on train: +0.0079 vs +0.0068 macro (dev: +0.0146 vs +0.0125, but `fit_years` is better without iota: +0.0083 vs +0.0071). By domain, the full extractor is better in politics (+0.015 over years-only on train) and cardano, while years-only is better in history, hsm and economics on train. The two variants were tuned separately (different τ, center, norm and b), so this is not a clean ablation.
- **"Present" helps on train** (+0.0016 over `fit`), mainly through iota (+0.041) and workplace (+0.005). No iota train query has a calendar time, so present expressions are the only time signal there. On dev it adds +0.0010 macro. The benefit is small and concentrated in small domains.

Example (train, history `63955_546`, `fit`): "On 15 October 1810 the 92nd Regiment were billeted in ruined houses at Crozendera, Portugal...". BM25's top 5 were documents whose extracted dates are all 1990–2019. They fall to ranks 28–60, and the gold documents (dated 1809–1812) move from rank 9 to 1 and from 11 to 4. nDCG@10 goes from 0.14 to 0.67.

## 5. Query temporal-type classifier (`classify.py`, optional)

- **Labels** come from `guidance_train.jsonl` only; `guidance_dev.jsonl` is never read. `is_temporal_query` is useless as a target (true for 96% of train queries), so the target is **calendar-anchored**: `is_temporal_query` and an `expected_granularity` that is calendar-like (year, decade, century, date, month, era, period, ...), as opposed to block heights, epochs and slots, durations, hours and minutes, versions, or no time. 85% of train queries are positive.
- **Model.** TF-IDF word 1–2 grams plus counts of extracted expression kinds plus domain one-hot, with class-balanced logistic regression (C = 3, seed 13). Train gate values are 5-fold out-of-fold predictions; dev values come from a model fit on all of train.
- **Quality.** CV AUC 0.882, accuracy 0.865 at 0.5. The majority class alone gives 0.852; the rule "extractor found a time" gives 0.713.
- **Effect.** None. Among queries with an extracted time (the only ones the score changes), 97% get p ≥ 0.5, so the hard gate almost never switches the score off, and the soft gate only shrinks b slightly. `fit_gated` ties `fit` on train (both +0.0079) and is lower on dev. The gate would only matter with a much noisier extractor, for example one firing on numbers in crypto domains. With the current context filters, the extractor already acts as its own gate.

## 6. Leakage hygiene and a disclosure

- At prediction time the rescorer sees only query ids and text, the input run, and documents. Qrels are used in `analyze.py` (train, description only) and in `tune.py` (train, as the tuning objective). The classifier uses `guidance_train.jsonl` labels only.
- Dev was scored once per variant after tuning. No dev number fed back into a parameter.
- **Disclosure.** After the first dev scoring, I looked at the dev queries whose score changed most (iota and cardano) and found an extraction bug: "25 ADA", "2 ADA" (Cardano amounts), "AD1" and hex fragments were parsed as era-marked years. I fixed the bug in general terms (strict uppercase era markers, no trailing letter or digit, a space required in "AD 79"), checked that the fix removes the same pattern from *train* candidates (cardano had 16 × "10 AD", economics 18 × "AD1"), bumped the extractor to v2, re-extracted, and **re-tuned everything on train**. The v1 numbers were: train `fit` 0.1170 / `fit_present` 0.1175; dev `fit` 0.1244 / `fit_present` 0.1310. On v1 the bug's fake years happened to help cardano on train (+0.010) and hurt it badly on dev (−0.066 for `fit`). The v1 rows were removed from `results/experiments.csv` (they were never committed). Treat the dev numbers as slightly optimistic for that reason, although the fix is a generic correctness fix and not a tuned parameter.

## 7. Applying it to another run (dense, hybrid)

```bash
export PYTHONPATH="$PWD/src" RETECO_DATA=data/data/track1_tempo
# candidates not yet in cache/temporal/doc_times are extracted on the fly
python -m reteco.temporal.rescore --run runs/dev/hybrid.trec --split dev \
    --config configs/temporal_fit_present.yaml
# -> runs/dev/hybrid+temporal_fit_present.trec, row appended to results/experiments.csv
```

The configs were tuned on BM25 candidates. Scores are normalised per query (z-score for `fit_present`, min-max for `fit`), so they transfer to other first-stage runs. b should still be re-tuned on the hybrid train run (`python -m reteco.temporal.tune --run runs/train/hybrid.trec`): a stronger first stage usually needs a smaller temporal weight. To precompute times for a run without scoring it: `python -m reteco.temporal.doctimes --runs runs/train/hybrid.trec runs/dev/hybrid.trec`.

## 8. Reproduce

```bash
export PYTHONPATH="$PWD/src" RETECO_DATA=data/data/track1_tempo
python -m reteco.temporal.doctimes --runs runs/train/bm25.trec runs/dev/bm25.trec  # ~6 min, cached
python -m reteco.temporal.analyze --run runs/train/bm25.trec                        # section 2
python -m reteco.temporal.classify                                                  # gate CV metrics
python -m reteco.temporal.tune --run runs/train/bm25.trec                           # writes configs/temporal_*.yaml
for v in boost fit fit_years fit_present fit_gated fit_gated_hard; do
  for s in train dev; do
    python -m reteco.temporal.rescore --run runs/$s/bm25.trec --split $s --config configs/temporal_$v.yaml
  done
done
python -m pytest tests/test_temporal.py
```

## 9. Limitations and next steps

- Document times are only year-level; month-level precision would matter for the crypto domains (fork dates), but few of their queries carry a time.
- The gazetteer is hand-written and English-only. Ambiguous names are guarded (e.g. "Reconstruction" is case-sensitive; Cardano era names need "era" or "hard fork"), but recall for local or regional events is low.
- Relative expressions attach only to an expression that immediately follows them ("before 1600", "after WWII"). "Ten years after the war" is not handled.
- The weight b was tuned globally. Per-domain b (the domain is known at prediction time) is the obvious next step, but with 7–35 train queries in small domains it would overfit; a two-group b (history-like vs crypto-like domains) would be a safer try.
- Re-tune b on the hybrid run once it exists (section 7).

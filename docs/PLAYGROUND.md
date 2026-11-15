# Playground: autonomous improvement loop on the final system

> **Merged into `main` on 2026-11-11.** The champion below is now the final system, `configs/final.yaml` (runs `runs/{train,dev}/final.trec`). The system the loop started from was renamed `final_v1` (`configs/final_v1.yaml`, `runs/{train,dev}/final_v1.trec`); the rest of this page uses the names from the loop, where "original `final`" means `final_v1` and `playground_final` means the new `final`. Build the two fine-tuned models first with `scripts/build_finetuned_models.sh`.

## Summary

An autonomous loop on branch `playground` tried **15 changes** to the final system (then `configs/final.yaml`, now `configs/final_v1.yaml`). It tuned only on a fixed 80% "tune" part of TRAIN, judged only on the 20% "holdout" part, and kept a change only if holdout nDCG@10 rose by ≥ 0.003 at ≤ 2× latency. **5 were kept and 10 rejected.** The loop stopped after 4.5 h: every backlog item was done, and the last two ideas were tried after three rejections in a row.

**Final champion: `playground_final`** (= `configs/pg_e20.yaml`; now `configs/final.yaml`). Tuned BM25 + bge-small, now with a **fine-tuned query encoder** (frozen document vectors) → RRF (k 10, w_dense 0.4) → **MiniLM cross-encoder fine-tuned on train qrels with hard negatives, on 128-token queries**, top 200 → interpolation (a 0.35, z-score) → the unchanged temporal and near-duplicate stages. Same latency as before: ~1110 ms per query, 96% of it the cross-encoder.

| | holdout nDCG@10 | dev nDCG@10 | dev w/o iota | dev R@100 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| original `final` (now `final_v1`) | 0.3155 | 0.3403 | 0.3105 | 0.7436 | 1101 |
| **`playground_final`** (now `final`) | **0.3671** | **0.3679** | **0.3457** | 0.7397 | 1110 |
| Δ | +0.0516 | **+0.0276** | +0.0353 | −0.0039 | |

**The one dev comparison** (scored once at the end; nothing was changed afterwards): +0.0276 macro nDCG@10, 95% CI [+0.010, +0.046] (domain-stratified paired bootstrap, 10k). 226 dev queries improve, 123 get worse, 170 are unchanged. It is better in 9 of 13 domains: history +0.076 (240 queries), genealogy +0.076, workplace +0.100, quant +0.079, hsm +0.050. It is worse in law −0.062 (11 queries), iota −0.064 (3 queries), politics −0.020 and bitcoin −0.019. The original system was re-run through the pipeline and reproduces its logged 0.3403 exactly.

**What helped: fine-tuning, and nothing else clearly.**

| kept | change | Δ holdout | Δ w/o iota |
| --- | --- | ---: | ---: |
| E1 | fine-tune the MiniLM cross-encoder (hard negatives from the hybrid top 200) | +0.0237 | +0.0383 |
| E5 | re-tune fusion k / w_dense with the reranker on top | +0.0054 | −0.0068 |
| E8 | fine-tune only the bge-small query encoder against frozen doc vectors (own idea) | +0.0060 | +0.0065 |
| E3 | cross-encoder retrained on 128-token queries (more room for the document) | +0.0074 | +0.0093 |
| E20 | both fine-tuned models retrained on all of tune with fixed hyperparameters | +0.0092 | +0.0086 |

- **The cross-encoder fine-tuning (E1) is the main result.** The MS MARCO reranker was the weak link in `docs/RERANK.md` (alone far below the first stage). Trained on ~800 train queries for 15 minutes, it improves alone by +0.06 on tune_val and lifts the whole system by +0.024 on holdout.
- **E5 is probably noise.** Its holdout gain comes entirely from iota's 2 holdout queries; without iota it is negative, and E19 later found the opposite fusion setting on tune_val, which lost on holdout. It was kept because the rule was fixed in advance (decision 11).
- **Rejected:** per-domain interpolation weight (E2), three-way RRF (E4), rerank depth 300 (E12), re-tuned temporal/dedup (E9), query shortening (E7), a cross-encoder ensemble (E14), fusion re-tuned again (E19), e5-small-v2 as a third retriever (E6, which cannot be encoded within the 2 h cap) and 64-token queries (E16). Every "re-tune a few weights" idea was rejected: with 194–243 queries those tweaks sit inside the noise, and the big temporal grid (E9) overfit outright.
- **Holdout overstated the gain**: +0.052 on holdout vs +0.028 on dev. That is expected after picking the 5 best of 15 tries on the same 243 queries, and iota's 2 holdout queries swing the macro by up to ±0.015 by themselves (E5, E14).
- **Caveat for reuse:** the champion needs two fine-tuned models that live in the gitignored `cache/playground/` (~90 MB each); `scripts/build_finetuned_models.sh` rebuilds both (the exact commands are also at the top of `configs/pg_e20.yaml` and `configs/pg_e20_dense.yaml`). A test rebuild changed holdout nDCG@10 by −0.0004 (see Rebuild check), which is the expected run-to-run variation.

## Setup

- Branch `playground` from `main` (bd11165). `main` is never modified, nothing is pushed. After the loop, the branch was merged into `main` and deleted; its commits are in `main`'s history (`ceb0f4f`..`5189fe6`).
- **Split** (`configs/playground_split.json`, `python -m reteco.playground.common --make-split`): TRAIN only, stratified per domain, seed 13. Holdout = max(2, round(0.2 n)) queries per domain, the rest is tune: **968 tune / 243 holdout**. Dev is not used during the loop.
- **Champion** = `configs/final.yaml` (now `configs/final_v1.yaml`). Its holdout score is the baseline.
- **Harness** (`python -m reteco.playground.harness --config <yaml> --name <run>`): runs the full pipeline (`reteco.pipeline.run_pipeline`) on train and scores macro nDCG@10 over the 13 domains on the tune and holdout queries separately. First-stage lists are cached under `cache/playground/first/`.
- **Keep rule**: holdout nDCG@10 ≥ champion + 0.003 and latency ≤ 2 × champion. Parameters tuned on tune only; holdout only judges.
- Log: `results/playground.csv` (one row per experiment) and the table below.

## Experiments

In run order. "tune" is the tune part, or tune_val (194 queries) where a model trained on tune_train is involved (decision 8). Holdout = 243 queries, macro over 13 domains. Δ is against the champion at that time. Latency is ms per query for the whole pipeline (M4, MPS). All rows are in `results/playground.csv`.

| id | experiment | tune | holdout | Δ holdout | holdout w/o iota | ms/query | min | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | baseline: `configs/final.yaml` (now `final_v1.yaml`) | 0.2957 | 0.3155 | | 0.2805 | 1100 | 15 | champion |
| E1 | fine-tuned MiniLM cross-encoder | 0.3306 (tv; was 0.2852) | **0.3392** | **+0.0237** | 0.3188 | 1113 | 46 | **KEPT** |
| E2 | per-domain interpolation weight | CV 0.2961 vs 0.2949 | 0.3349 | −0.0043 | 0.3142 | 1112 | 20 | rejected |
| E4 | three-way RRF instead of interpolation | 0.3333 (tv) | 0.3406 | +0.0014 | 0.3189 | 1111 | 25 | rejected |
| E12 | rerank depth 100 / 200 / 300 | d300 = d200 ± 0.001 | not scored | | | | 10 | rejected (tuning kept 200) |
| E9 | re-tune temporal + dedup on top | 0.3384 (tv) | 0.3324 | −0.0068 | 0.3115 | 1112 | 20 | rejected |
| E5 | re-tune fusion k / w_dense | 0.3336 (tv) | **0.3446** | **+0.0054** | 0.3121 | 1114 | 30 | **KEPT** (iota only, see below) |
| E8 | fine-tuned query encoder, frozen doc vectors | 0.3399 (tv) | **0.3505** | **+0.0060** | 0.3185 | 1120 | 55 | **KEPT** |
| E7 | extractive query shortening, first stage | full query best | not scored | | | | 25 | rejected (tuning kept full) |
| E3 | cross-encoder on 128-token queries (retrained) | 0.3443 (tv) | **0.3579** | **+0.0074** | 0.3279 | 1105 | 50 | **KEPT** |
| E14 | ensemble original + fine-tuned cross-encoder | 0.3410 (tv) | 0.3594 | +0.0015 | 0.3281 | ~2160 | 20 | rejected |
| E19 | re-tune fusion again after E8 | 0.3499 (tv) | 0.3511 | −0.0068 | 0.3220 | 1100 | 15 | rejected |
| E6 | e5-small-v2 as a third retriever (small domains) | small: 0.3361 vs 0.3276 | small: 0.3326 vs 0.3433 | −0.0107 (small) | | | 35 | rejected (cap) |
| E16 | cross-encoder on 64-token queries (retrained) | interp 0.3309 vs 0.3419 | not scored | | | | 45 | rejected (tuning kept 128) |
| E20 | retrain both fine-tuned models on all of tune | n/a | **0.3671** | **+0.0092** | 0.3365 | 1109 | 35 | **KEPT** (final champion) |

Total time in experiments: about 4.5 h (20:59–01:30), plus the dev runs.

### Notes per experiment

- **E1, fine-tuned cross-encoder (backlog 1).** The single big win. MS MARCO MiniLM alone was far below the first stage (`docs/RERANK.md`). Fine-tuned on 774 tune_train queries for one epoch, it scores 0.2461 alone on tune_val (was 0.1846) and the interpolated run 0.3304 (was 0.2881). Holdout +0.0237, 95% CI [+0.002, +0.050], 104 queries better / 33 worse. Without iota the gain is +0.038. Epoch 2 fit tune_train better (loss 1.23 vs 1.70) but was no better on tune_val (0.3302 vs 0.3304) and worse alone. Latency is unchanged: same architecture.
- **E2, per-domain a (backlog 2).** Shrinkage toward the global a was chosen by 5-fold CV inside the tuning set. On the original cross-encoder the CV preferred the global a outright (m = ∞: 0.2891 vs 0.2747 for unshrunk per-domain a). On E1 it picked a weak shrinkage (m = 10, CV +0.001), which lost 0.0043 on holdout. There aren't enough queries per domain to estimate a per-domain weight.
- **E4, three-way RRF (backlog 4).** Below linear interpolation at the interpolation stage on both tune (original CE: 0.2837 vs 0.2890) and tune_val (E1: 0.3272 vs 0.3304). After temporal and dedup it was slightly higher on tune_val, and +0.0014 on holdout: under the bar.
- **E12, depth.** Depth 300 tied depth 200 on tune_val (0.3309 vs 0.3304) at 1.5× latency; depth 100 lost 0.004. The tie rule from `docs/RERANK.md` keeps 200.
- **E9, re-tune temporal and dedup.** A large grid (thousands of temporal settings) on 194 queries: +0.008 in-sample, −0.007 on holdout (quant 0.314 → 0.214). The clearest overfitting case of the loop.
- **E5, fusion re-tuned with the reranker on top (backlog 5, fusion part).** Kept by the rule, but see decision 11: the whole gain is iota's 2 queries (+0.151), the other 12 domains lose 0.0068 on average, and more queries got worse (57) than better (39). E19 later re-tuned fusion on top of E8 and E3, chose the original k = 20 / w = 0.5, and lost 0.0068 on holdout. So both directions of this change "lose" on one set. My reading: fusion weights in this range are inside the noise of both sets. Sparse k1/b were not re-tuned: a per-domain grid needs new cross-encoder pairs for every setting, and the sparse tuning in `docs/SPARSE.md` was already done on all of train.
- **E8, query-encoder fine-tuning (own idea).** Only the query tower of bge-small is trained, against the frozen cached document vectors, so no corpus re-encoding (2.4 h) is needed and every hard negative is a precomputed vector. It took 3 min per epoch. First-stage hybrid nDCG@10 on tune_val went 0.2771 → 0.3020 (epoch 2); the full system on holdout +0.0060, +0.0065 without iota. Per query it is mixed (62 better / 68 worse): history (112 holdout queries) lost 0.026 while law (+0.077), workplace (+0.048) and monero (+0.034) gained.
- **E7, query shortening (backlog 7).** 34 variants, none better than the full post on tune_val, in nDCG@10 or in Recall@200. Keeping only the title and the first sentence costs about 0.07. Consistent with `docs/SPARSE.md`, where long queries with a large k1 worked best.
- **E3, cross-encoder input (backlog 3).** A 128-token query cap (the first ~100 words: title and opening) leaves ~380 tokens of each document instead of ~250. Retrained with the same recipe: CE alone on tune_val 0.2497 vs 0.2235, holdout +0.0074 (+0.0093 without iota). E16 tried 64 tokens: clearly worse on tune_val, so 128 is the sweet spot among those tried. Title-only and document-window variants were not run (decision 15).
- **E14, cross-encoder ensemble.** +0.0015 on holdout at ~2× latency.
- **E20, retrain on all of tune.** With epochs and weights fixed from E3/E8, the cross-encoder and query encoder were retrained on all 968 tune queries (25% more than tune_train). Holdout +0.0092 (73 better / 47 worse), with no tuning involved, so it is the cleanest of the later gains.
- **E6, second dense model (backlog 6).** e5-small-v2 encodes at 170 docs/s on this machine, so the full corpus would take 1.9 h, plus cross-encoder scoring and tuning: over the 2 h cap. On the 5 small domains (first stage only) it hurt holdout (0.3326 vs 0.3433). e5 alone scores 0.106 there vs 0.348 for bge with the fine-tuned query encoder. I checked the embeddings (fp16 matches fp32, mean pooling, no NaN): e5-small is just weak on long posts.

## Reproduce

```
source .venv-dense/bin/activate && export PYTHONPATH="$PWD/src"
python -m reteco.playground.harness --config configs/final_v1.yaml --name pg_champion0      # E0 on tune + holdout
# E20 models (= scripts/build_finetuned_models.sh; about 25 min on an M4)
python -m reteco.playground.finetune_ce --config configs/final_v1.yaml --name minilm_ft_q128_all \
    --epochs 2 --n-neg 7 --groups-per-step 4 --max-query-tokens 128 --train-on tune
python -m reteco.playground.finetune_qenc --name bge_q_all --epochs 2 --train-on tune
python -m reteco.playground.harness --config configs/pg_e20.yaml --name pg_e20 --parts holdout
python -m reteco.playground.compare runs/train/pg_e3.trec runs/train/pg_e20.trec          # paired holdout comparison
python -m reteco.pipeline --config configs/final.yaml --splits dev                       # the dev run (done once, as playground_final)
```

The models of the intermediate champions (E1, E3, E8) are rebuilt with the commands at the top of their configs. Scripts for every experiment, kept or rejected, are in `src/reteco/playground/`.

## Rebuild check (after the merge)

`scripts/build_finetuned_models.sh` was run once from scratch (31 min on the M4) to test whether the champion's models can be rebuilt. The canonical models were backed up first and restored afterwards. The rebuilt copies were scored under a separate path (`cache/playground/rebuild/`), because the cross-encoder score cache is keyed by model path. Holdout only; dev was not scored with the rebuilt models.

| | max abs weight diff vs canonical | holdout nDCG@10 | w/o iota |
| --- | ---: | ---: | ---: |
| canonical models (logged `playground_final` / `final`) | | 0.3671 | 0.3365 |
| rebuilt with the script | CE 3.4e-5, query encoder 7.7e-6 | 0.3667 | 0.3360 |
| Δ | | −0.0004, 95% CI [−0.0011, 0.0000] | −0.0004 |

- **The rebuild is not bit-identical.** MPS training is not deterministic even with a fixed seed. The query encoder's per-epoch tune_val scores matched the original log to four decimals.
- **The rebuild nearly matches.** 240 of 243 holdout queries score the same. The other three move slightly (1 better, 2 worse), a net loss in history (−0.002) and workplace (−0.003).
- **Expected run-to-run variation is therefore about ±0.001 nDCG@10.** That is a tenth of the +0.003 keep threshold and far below the measured gains.
- **The canonical models remain the reference.** With them restored, `configs/final.yaml` reproduces dev 0.3679, and `runs/dev/final.trec` is byte-identical to the logged run.

## Ablation of the final system (dev, `results/ablation_final.csv`)

Leave-one-out ablation of `final` (= `playground_final`), run after the loop with `python -m reteco.pipeline --config configs/final.yaml --ablation-only`. Each row changes one thing and keeps every other setting fixed, as in the `final_v1` ablation (`docs/RERANK.md` section 6, now `results/ablation_final_v1.csv`). The two last rows swap a fine-tuned model back to the original one. Scored once on dev; **nothing was changed because of it**.

| Removed | dev nDCG@10 | Δ vs final | Δ w/o iota | dev R@100 | ms/query | Δ in `final_v1` ablation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nothing (`final`) | 0.3679 | | (0.3457) | 0.7397 | 1115 | |
| sparse retrieval (dense only) | 0.3046 | −0.0633 | −0.0658 | 0.6035 | 1103 | −0.0321 |
| dense retrieval (sparse only) | 0.3502 | −0.0177 | −0.0160 | 0.7125 | 1113 | −0.0440 |
| cross-encoder (fusion → temporal → dedup) | 0.3171 | −0.0508 | −0.0535 | 0.7102 | 68 | −0.0140 |
| interpolation (cross-encoder score only) | 0.2651 | −0.1028 | −0.0837 | 0.7226 | 1111 | −0.0759 |
| temporal rescorer | 0.3685 | +0.0005 | +0.0006 | 0.7397 | 1111 | +0.0003 |
| query cleaning for the cross-encoder | 0.3617 | −0.0062 | −0.0067 | 0.7423 | 1153 | −0.0055 |
| near-duplicate demotion | 0.3617 | −0.0063 | −0.0067 | 0.7397 | 1088 | −0.0100 |
| cross-encoder fine-tuning (original ms-marco MiniLM, 128-token queries) | 0.3447 | −0.0232 | −0.0283 | 0.7424 | 1124 | |
| query-encoder fine-tuning (original bge-small, `dense_best`) | 0.3727 | **+0.0048** | +0.0021 | 0.7612 | 1109 | |

Paired comparisons with `final` for the two fine-tuning rows (domain-stratified bootstrap, 10k; randomization test):

- **The fine-tuned cross-encoder holds up on dev.** Swapping in the original loses 0.0232, 95% CI [−0.038, −0.009], p = 0.004 (without iota, CI [−0.044, −0.012]); 216 dev queries get worse and 99 better. It is the one playground change confirmed on its own on dev.
- **The fine-tuned query encoder does not.** The original bge-small query encoder scores +0.0048 *higher*, CI [−0.006, +0.016], p = 0.44 (without iota +0.0021, CI [−0.010, +0.014]); 151 queries better, 130 worse. The difference is within noise either way, but the holdout gains of E8 (+0.0060) and of its part in E20 do not carry over, and recall is lower with the fine-tuned encoder (R@100 0.7397 vs 0.7612). It stays in `final`: removing it now would be a choice made on dev.
- **The fine-tuned cross-encoder shifted the weight of the stages.** The cross-encoder is now worth +0.051 (was +0.014 in `final_v1`), but its score alone is still far below the combination (interpolation −0.103). Sparse retrieval matters more (−0.063 vs −0.032) and dense less (−0.018 vs −0.044), in line with E5's lower dense weight in the fusion (w_dense 0.4, k 10).
- **The other components behave as in `final_v1`.** Dedup (+0.006) and query cleaning (+0.006) help a little. Temporal is again the only component whose removal helps (+0.0005).
- Each variant is logged in `results/experiments.csv` as `final_no_*` dated 2026-11-14. The `final_v1` ablation used the same names before the rename (dated 2026-11-10).

## Decisions made

1. **At least 2 holdout queries per domain.** With round(0.2 n), iota (7 train queries) would have had one holdout query, which alone moves the macro by up to 1/13. It has 2 (so 2/7 instead of 1/7); every other domain is exactly round(0.2 n).
2. **The champion was tuned on all of train, including holdout.** Its holdout score (0.3155) is therefore slightly optimistic, while every challenger is tuned on tune only. This biases the loop toward rejecting. I did not re-tune the champion on tune, because the brief defines the champion as `configs/final.yaml` (now `final_v1.yaml`) as it is.
3. **Holdout without iota is reported next to the macro** as a diagnostic (iota has 2 holdout queries, and the champion scores 0.735 on them). Decisions use the macro, as the rule says.
4. **Order: E1 (fine-tuned cross-encoder) first.** It has the largest expected gain, and it changes the cross-encoder scores that E2–E5 build on. E2 and E4 were run once on the original cross-encoder while E1 was training (CPU only, cached scores); they are re-run on top of the champion when the champion changes.
5. **Playground runs are not written to `results/experiments.csv`.** They are scored on parts of train, which that file has no column for; they go to `results/playground.csv`. Only the two dev runs at the end are logged there. Playground run files (`runs/train/pg_*`) are gitignored.
6. **Latency** is the pipeline's measured total per query over the queries that were run (cross-encoder time from the measured per-call timings in the score cache, as in `docs/RERANK.md`).
7. **Temporal and dedup configs stay fixed** when an earlier stage changes. Re-tuning them on top is a separate experiment (E9).
8. **Models trained on tune are validated on a split of tune.** E1 and E8 train on `tune_train` (80% of tune per domain, seed 13, `reteco.playground.common.subsplit_tune`) and choose the epoch and interpolation weight on `tune_val` (the other 20%, 194 queries). After E1, every parameter that depends on the cross-encoder's scores is tuned on tune_val only, because its scores on tune_train are in-sample. For E1, E4, E5, E8 and later the "tune" column is the tune_val score.
9. **E1 fine-tuning hyperparameters were fixed a priori, not tuned**: softmax cross-entropy over 1 positive + 7 hard negatives, lr 2e-5 with 10% warmup and linear decay, 4 groups (32 pairs) per step, 2 epochs, fp32 on MPS. 7 negatives instead of 15 to keep an epoch near 15 minutes. Negatives within word 5-gram Jaccard ≥ 0.6 of a positive (or identical text) were dropped (304 of them), so unjudged copies of gold documents are not trained as negatives.
10. **When tuning selects the champion's own setting, holdout is not scored** (E7, E12). There is no change to judge, and scoring it anyway would only invite choosing by holdout.
11. **E5 was kept although its gain is all iota.** Holdout +0.0054 passes the rule, but without iota it is −0.0068 and 57 holdout queries got worse vs 39 better. Decision 3 fixed the macro as the criterion before any result, and overriding it after seeing a result would make the rule selective, so it was kept and flagged. It is the weakest of the kept changes.
12. **Rejected experiments are reverted in the system code** (pipeline hooks for per-domain a, RRF3 and query shortening were removed), but their playground scripts stay under `src/reteco/playground/` as a record, and their configs are moved to `cache/playground/configs/` (not committed).
13. **Fine-tuned models live in `cache/playground/` (gitignored, ~90 MB each)**. The configs that use them name the exact command that rebuilds them.
14. **Latency on a subset of queries.** When only holdout is run, first-stage and fusion time (computed for all train queries) is rescaled to the queries run. Latencies printed during tune_val grids are not comparable: tune_val pairs were partly scored while a training job shared the GPU, and incremental scoring of a few new pairs per call inflates per-pair time. Kept/rejected decisions use holdout runs only.
15. **Backlog 3 was narrowed to query length.** With a fine-tuned cross-encoder, an input variant is only a fair test after retraining (~30 min plus scoring). I ran the variant with a clear mechanism (128 tokens, more room for the document) and its continuation (64), not title-only or alternative document windows.
16. **Order of the remaining backlog.** Done roughly by expected gain per hour: E1 → cheap tuning of interpolation (E2, E4, E12, E9) → fusion (E5) → own idea E8 → E7 → E3 → E14, E19 → E6 → E16 → E20. Two of my own ideas (E8, query-encoder fine-tuning; E3's retrain approach) gave real gains; the cheap re-tuning ideas did not.
17. **Continuing past 3 rejections in a row.** After E14, E19 and E6 were rejected I still had two ideas built on the only mechanism that had worked (fine-tuning): E16 (shorter query) and E20 (retrain on all of tune). I ran those and then stopped, at about 4.5 h of the 10 h budget.
18. **E20 fixes every hyperparameter in advance** (epochs, a, norm, query length from E3/E8) because training on all of tune leaves no tune-internal validation set. It is judged on holdout like everything else.
19. **Concurrent GPU jobs.** Two training jobs at once on the 16 GB M4 pushed the machine into swap (12.7 GB) and slowed both several-fold. After that, trainings ran one at a time.
20. **Commits follow CLAUDE.md**: short plain messages, no AI attribution or Co-Authored-By lines, nothing pushed, `main` untouched.

## Round 2

A second autonomous loop on branch `playground2` (from `main` at `da11bd9`), starting from the round-1 champion `configs/final.yaml`. Same split (`configs/playground_split.json`: 968 tune / 243 holdout), same harness and the same keep rule: holdout nDCG@10 ≥ champion + 0.003 and latency ≤ 2 × champion. Models are trained on tune (tune_train / tune_val where a validation set is needed), hyperparameters are fixed in advance where possible, dev is not touched until the end, and nothing is pushed. Experiments are logged in `results/playground.csv` with ids `R2-E<n>`, where `<n>` is the backlog item (0 = baseline). Configs are `configs/pg2_*.yaml`; run files are `runs/train/pg2_*.trec` (not committed).

### Experiments

| id | experiment | tune | holdout | Δ holdout | holdout w/o iota | ms/query | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R2-E0 | baseline: `configs/final.yaml` (round-1 champion) | 0.4322 (in-sample) | 0.3671 | | 0.3365 | 1112 | champion |
| R2-E3 | fusion reverted to round-1's original (k 20, w_dense 0.5), i.e. without E5 | n/a | **0.3723** | **+0.0052** | 0.3434 | 1111 | **KEPT** (simpler setting) |
| R2-E1 | MiniLM-L-12 cross-encoder, round-1 recipe | n/a | 0.3473 | −0.0250 | 0.3356 | 2084 | rejected |
| R2-E2 | cross-encoder with 15 hard negatives (was 7) | n/a | 0.3638 | −0.0085 | 0.3342 | 1087 | rejected |

### Notes per experiment

- **R2-E0, baseline.** `configs/final.yaml` reproduces round-1 E20 exactly on holdout (0.3671). Its tune score is in-sample: both fine-tuned models were trained on tune. Latency 1112 ms/query on the holdout run, so the latency cap is 2223 ms.
- **R2-E3, revert E5 (backlog 3).** No tuning: the champion with fusion set back to k 20 / w_dense 0.5. Holdout 0.3723 vs 0.3671, so the champion *with* E5 is 0.0052 worse rather than ≥ 0.003 better, and the simpler original setting is preferred (it also passes the keep rule by itself). Paired: CI [−0.017, +0.027], 56 better / 50 worse. workplace (+0.169 over 5 queries) carries most of it. Together with round 1 (E5 +0.0054, E19 −0.0068) this is the third holdout swing of ±0.005–0.007 from the same two fusion settings: they are within noise, and the original is kept for being the setting chosen on all of train before the playground.
- **R2-E1, MiniLM-L-12 (backlog 1).** Latency probe first (30 holdout queries, top 200, two rounds): the L-12 cross-encoder takes 2080–2165 ms per query vs 1029–1098 for L-6, so the whole pipeline would be about 2170 ms, 1.95× the champion, just under the cap. Recipe fixed in advance as E20's (all of tune, epoch 1 of a 2-epoch schedule, 7 hard negatives from `final_v1`'s first stage, 4 groups per step, 128-token queries, a 0.35 z-score), with gradient accumulation because 32 pairs × 512 tokens did not fit in MPS memory. Holdout **0.3473, −0.0250** (CI [−0.051, +0.003], 74 better / 64 worse) at 2084 ms per query (1.88×). Half of the loss is iota (−0.230 over its 2 queries), but without iota it is still −0.0079: law (+0.060), politics (+0.028) and history (+0.017) gain, workplace (−0.112), cardano (−0.036), genealogy (−0.034) and monero (−0.029) lose. Its training loss was lower than L-6's under the same recipe (epoch-1 mean 1.528 vs 1.609), so the extra capacity fits tune better without ranking holdout better. With ~1 000 training queries the 6-layer model is enough. Rejected. It took three training attempts (decision 7).
- **R2-E2, 15 negatives (backlog 2).** L-6 retrained with 15 hard negatives per positive (16 documents per group), all else as E20. Holdout **0.3638, −0.0085** (CI [−0.029, +0.009]), −0.0092 without iota. Per query it looks better (66 better / 37 worse; history +0.021 over 112 queries, cardano +0.029), but the small domains lose (workplace −0.090 on 5 queries, law −0.030, quant −0.025) and the macro decides. Rejected. As with E14 in round 1, a different training signal for the same 6-layer model mostly reshuffles which queries win.

### Decisions made

1. **Backlog order from the round-1 dev ablation.** The ablation of the round-1 champion (above, `results/ablation_final.csv`) showed that the fine-tuned cross-encoder's gain held up on dev while the fine-tuned query encoder's did not, so the backlog starts with the cross-encoder line of work. Dev results were used only to order the backlog, never for a keep/reject decision; the keep rule on holdout is unchanged.
2. **Backlog item 3 was run first.** It needs no training (about 5 minutes) and decides the fusion setting that items 1 and 2 are judged on, so running it first avoids judging the cross-encoder variants under two fusion settings.
3. **R2-E3's rule.** The brief's rule for item 3 (prefer the original setting unless the champion with E5 is better by ≥ 0.003) and the keep rule point the same way here, so no conflict arose.
4. **Hyperparameters fixed in advance for the cross-encoder items.** R2-E1 and R2-E2 use E20's recipe unchanged (training on all of tune, epoch 1 of a 2-epoch schedule, interpolation a 0.35 z-score) and change one thing each, the base model or the number of negatives. Nothing is selected on tune_val, since round 1 showed that small re-tunes are noise.
5. **Gradient accumulation for L-12.** The first L-12 smoke test ran out of MPS memory at 4 groups × 8 pairs × 512 tokens in fp32. `finetune_ce.py --micro-groups` splits a step into smaller forward passes and sums the loss over the step's groups, so the update is the same as one pass over 4 groups.
6. **Budget clock.** The 12 h budget counts active time. While a question to the user was open (17:21–00:03) the loop was blocked and that time is not counted.
7. **L-12 needed three training attempts; ~3.3 h of wall time, over the 3 h cap.** (a) The smoke test (100 steps on small domains) worked but hung on interpreter exit for 25 min and was killed. (b) The full run with 2 groups per forward pass was stopped by Claude Code at step ~110 because the machine was critically low on memory (swap 10.7 of 12 GB). On the user's instruction it was retried with 1 group (8 pairs) per forward pass and the in-memory document texts cut to the 4,000 characters training reads. (c) That run was stopped again at step ~800 of 874, before its checkpoint. The cause was PyTorch's MPS cache: wired memory grew to ~11 GB with variable input shapes and fell to 2.4 GB once the job stopped. (d) On the user's instruction the third run released the MPS cache every 50 steps and saved a resumable state every 200 steps (`finetune_ce.py --resume-every`). Wired memory stayed at 4–5 GB and epoch 1 was saved at 00:39. The user set a deadline of 01:15 for that checkpoint, with the instruction to reject L-12 as over the time cap if it was missed. It was met, so L-12 was judged on holdout and rejected on the score. These fixes (cache release, resumable state) are used for every later training job.
8. **R2-E2 memory handling.** The first R2-E2 launch (2 groups = 32 pairs per forward pass) reached 11.3 GB of wired memory by step 100, the level at which the L-12 runs were stopped. I stopped it myself and restarted with 1 group (16 pairs) per forward pass. At step 200 I resumed from the saved state with the MPS cache released every 10 steps instead of 50 (`--cache-every`). Neither change alters the update. It finished without a memory stop; wired memory moved between 3 and 8 GB.

### Resume point

Branch `playground2`. Champion: **R2-E3** (`configs/pg2_e3.yaml`, holdout 0.3723, 1111 ms/query). Done: R2-E0, R2-E3 (kept), R2-E1 (rejected). Done also: R2-E2 (rejected; 2 rejections in a row). In progress: R2-E4 (local LLM rewrites). Next: backlog items 4 (local LLM rewrites; Qwen2.5-1.5B/0.5B-Instruct are in the HF cache, `src/reteco/playground/llm_rewrite.py` and the `rewrite` hook in `reteco.pipeline` are written but not yet run), 5 (`src/reteco/playground/finetune_bienc.py`, written, not run) and 6.

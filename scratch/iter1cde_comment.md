🤖 **iter-1c/d/e: pair-aware deep-dive + wide-C grids. Iter-1 BFS was badly under-tuned on C.**

Three local follow-up experiments (no cluster). All hit the iter-0 cache directly.

### Three things to know up front

1. **New mendelian champion**: `logreg_l2(embed_last_l2, minus_llr)` with proper C tuning hits macro PA **0.7750 ± 0.018** — beats rank-mean (0.7682), zero-shot (0.7408), and the best iter-1 BFS cell (`sym_concat × xgboost` 0.7069). Pair-aware logreg on the same 2-scalar recipe ties at 0.7728.
2. **Iter-1 BFS pair-aware on high-D was crippled by C=1.0.** Re-running `mendelian × sym_concat × pairwise_logreg` with a wide C grid lifts it from 0.5235 (iter-1 BFS) to **0.7164** (iter-1e, optimum at C=1e-4). +0.193 macro PA from a single hparam fix.
3. **Pair-aware ≈ standard LogReg in the low-dim regime where they share the same well-tuned C.** Within ±0.005 macro on every dataset / scalar set tested. On the high-D regime, only pair-aware was tested with wide C — std LogReg on `sym_concat` with wide C is still an open data point.

### Iter-1c — every iter-1 classifier on scalars-only (5 features: `embed_last_l2`, `minus_llr`, `abs_llr`, `pooled_l2`, `pooled_cosine_dist`)

BFS C-grid (one value per knob), chrom-grouped 3-fold OOF.

| classifier | mendelian | complex | eqtl |
|---|---:|---:|---:|
| `pairwise_logreg` | **0.7631** | 0.5897 | **0.5296** |
| `logreg_l2` | 0.7625 | **0.5985** | 0.5193 |
| `linearsvc` | 0.7602 | 0.5974 | 0.5139 |
| `knn` | 0.7275 | 0.5976 | 0.5073 |
| `xgboost` | 0.6600 | 0.5424 | 0.4742 |

Switching from high-D pool blocks (iter-1) to 5 scalars flips the classifier ranking. **Linear methods now win across the board**; XGBoost tanks (overfits 5 features × ~5K train rows per fold). Pair-aware ties std LogReg on mendelian/eqtl, lags on complex. Cf. iter-1 BFS where high-D had KNN best on average (0.579) and pair-aware mid-pack (0.552).

### Iter-1d/e — wide-C grids on the same scalar sets

C swept over `logspace(-10, 0, 11)`. Selected C values printed per fold.

**Mendelian × `embed_last_l2 + minus_llr`** (`vs. rank-mean = 0.7682`):

| classifier | macro PA | selected Cs (one per outer fold) |
|---|---:|---|
| `logreg_l2` xwide | **0.7750** | `[1e-3, 1e-10, 1e-5]` (some folds hit lower boundary) |
| `pairwise_logreg` xwide | 0.7728 | `[0.1, 0.1, 1e-4]` (mixed) |

**Mendelian × `embed_last_l2 + minus_llr + abs_llr`**: 3-scalar doesn't help — std logreg drops to 0.7589, pair-aware 0.7534. Adding `abs_llr` is dilution on mendelian.

**Complex_traits × `embed_last_l2 + minus_llr`**: best supervised is std logreg xwide = **0.6459**, still **below zero-shot `embed_last_l2` 0.6541**. Pair-aware lags at 0.6178. Adding `abs_llr` doesn't help either.

**Eqtl × `embed_last_l2 + minus_llr + abs_llr`**: std logreg xwide = 0.5296, pair-aware xwide = 0.5293, **both still below zero-shot 0.5360**. Effectively tied.

**Mendelian × `sym_concat` (D=3840), pair-aware xwide**: macro PA = **0.7164** ± 0.020, selected C = `1e-4` in every fold (consistent optimum, not at grid boundary in this wider grid). +0.193 over iter-1 BFS (0.5235), but still below zero-shot 0.7408 — high-D supervised hits the same ceiling once C is properly tuned, just from a different direction.

### Why iter-1's BFS missed this

The BFS spec was a single ``C=1.0`` per classifier. For these features and sample sizes (D ≈ 2-6K, n_train ≈ 5-6K per outer fold, balanced class weights), the genuine optimum sits near `C ≈ 1e-3` to `1e-6` — heavy ridge regularization. C=1.0 was 3-6 orders of magnitude too loose, so the iter-1 BFS supervised cells were systematically under-fit (paradoxically, by being too high-capacity).

Memory I'm taking from this: **don't put a single C value in a BFS grid for sklearn-style ridge classifiers** — at minimum use 3 values spanning ~6 orders of magnitude (e.g., `[1e-6, 1e-3, 1e0]`); the boundary-flag tells you when to widen further.

### Updated supervised champions per dataset

| dataset | best recipe (so far) | macro PA | Δ vs zero-shot `embed_last_l2` |
|---|---|---:|---:|
| `mendelian_traits` | **`logreg_l2(embed_last_l2, minus_llr)` wide C** | **0.7750** ± 0.018 | **+0.034** |
| `complex_traits` | zero-shot `embed_last_l2` (untrained) | **0.6541** ± 0.032 | (ceiling) |
| `eqtl` | zero-shot `embed_last_l2` (untrained) | **0.5360** ± 0.022 | (ceiling) |

The mendelian gain is the only supervised-over-zero-shot win that survives wide-C tuning. Complex and eqtl ceilings still stand at frozen-`exp166-p1B`'s `embed_last_l2`.

### Pair-aware specifically — net assessment

* **In the low-dim regime** (2-5 scalars), pair-aware logreg is **statistically indistinguishable from standard logreg** when both have proper C tuning. The pair-difference loss optimises the right metric, but on these features standard binary-CE recovers an equivalent linear classifier.
* **In the high-D regime** (`sym_concat` D=3840), pair-aware **was clearly under-tuned in iter-1 BFS** (0.524 → 0.716 with wide C). Whether std logreg would close the same gap with wide C is the obvious next test — if it does, pair-aware isn't doing anything std logreg can't.
* The pair-aware machinery (`build_pairwise_diff_dataset` + `fit_pairwise_linear_weights` + `pairwise_oof_predict` in [`src/bolinas/supervised/classifiers.py`](https://github.com/Open-Athena/bolinas-dna/blob/436185b441e36e452873c325d989e2385338aee5/src/bolinas/supervised/classifiers.py)) is still the right primitive for iter-2 LoRA — the ranking loss directly optimises PairwiseAccuracy and avoids the class-prior calibration issue that BCE has.

### Code

* iter-1c: [`scratch/iter1c_scalars_only.py`](https://github.com/Open-Athena/bolinas-dna/blob/436185b441e36e452873c325d989e2385338aee5/scratch/iter1c_scalars_only.py)
* iter-1d: [`scratch/iter1d_pairaware_deepdive.py`](https://github.com/Open-Athena/bolinas-dna/blob/436185b441e36e452873c325d989e2385338aee5/scratch/iter1d_pairaware_deepdive.py)
* iter-1e: [`scratch/iter1e_widen_C.py`](https://github.com/Open-Athena/bolinas-dna/blob/436185b441e36e452873c325d989e2385338aee5/scratch/iter1e_widen_C.py)

Each ran in 2-15 min locally; total ~25 min including the slow high-D pair-aware fits.

### Next

* **Cheap (15-30 min local)**: std logreg with wide C on the high-D recipes that iter-1 BFS ran (sym_concat, abs_delta, mean_delta, etc.) to test whether pair-aware has a residual advantage at high-D or whether C-tuning alone closes the gap.
* **Cheap**: re-run a couple of iter-1's KNN/XGBoost cells with refined hparams on the top-performing recipes (already known to be mediocre but worth one more pass at proper tuning).
* **Iter-2 (real work)**: LoRA fine-tuning of `exp166-p1B` with the pair-aware ranking loss. Frozen-embedding ceiling on complex/eqtl (0.6541 / 0.5360) is the binding constraint and only a learned backbone can plausibly raise it.

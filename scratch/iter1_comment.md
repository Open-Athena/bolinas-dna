🤖 **iter-1 BFS sweep: 82 (dataset × recipe × classifier) supervised combos vs. 3 zero-shot baselines.**

Headline: a frozen-embedding sklearn head on `exp166-p1B`'s mean-pooled features (chrom-grouped 3-fold CV, BFS hparams = one value per knob) **does not beat the zero-shot `embed_last_l2` baseline on 2 of 3 datasets**.

### Best-of-everything per dataset (macro PA)

| dataset | best supervised | macro PA | zero-shot best (baseline) | macro PA | Δ |
|---|---|---:|---|---:|---:|
| `mendelian_traits` | `sym_concat × xgboost` | 0.7069 ± 0.020 | `embed_last_l2` | **0.7408** ± 0.019 | **−0.034** |
| `complex_traits` | `abs_delta_plus_scalars × knn` | 0.5760 ± 0.034 | `embed_last_l2` | **0.6541** ± 0.032 | **−0.078** |
| `eqtl` | `sym_concat × knn` | **0.5781** ± 0.022 | `embed_last_l2` | 0.5360 ± 0.022 | **+0.042** |

Global PA (single number across all pairs) tracks the same pattern: mendelian −0.004, complex −0.018, eqtl +0.015.

### Per-classifier mean macro PA across recipes×datasets

```
classifier         mean    max    n
knn              0.5792  0.6686   17
xgboost          0.5720  0.7069   17
pairwise_logreg  0.5519  0.6285   17
logreg_l2        0.5387  0.6193   17
linearsvc        0.5329  0.6179   14  (3 mendelian × big-feature cells skipped, see footnote)
```

KNN is the strongest classifier on average — non-parametric distance prediction edges out everything else. XGBoost wins the single best cell (mendelian). The linear classifiers (logreg, linearsvc, pairwise_logreg) underperform.

### Mendelian symmetric vs asymmetric (the directionality probe)

| | best (mendelian) | best classifier |
|---|---:|---|
| Symmetric features (`abs_delta`, `prod`, `sym_concat`, `traitgym_innerprod`, `abs_delta_plus_scalars`) | **0.7069** | `sym_concat × xgboost` |
| Asymmetric features (`mean_delta`, `mean_concat_plus_llr`) | 0.6838 | `mean_delta × xgboost` |

**Knowing the ref/alt direction doesn't help here.** The supervised head extracts at most as much signal from symmetric features as from asymmetric ones, even on the dataset (mendelian) where direction is biologically defined. Either the directional info is already encoded in the symmetric features (e.g. `sym_concat` carries both "where" and "how much"), or the classifier can't use direction efficiently from these embeddings.

### Why does zero-shot win? (working hypothesis)

The zero-shot `embed_last_l2` is a **scalar**: the flattened-sequence Euclidean distance between ref and alt embeddings. It's already a near-optimal compression of "how different are these two embeddings." A supervised head on `mean_ref/mean_alt/innerprod` (D=1920 each block) has ~6k feature dims vs. ~6.5k training rows per outer fold — high-D ill-conditioning, especially under the chrom-grouped split where each fold's train and test are *different chromosomes*. The supervised head pays for its extra capacity without recovering more signal than the scalar already encodes. KNN — which doesn't fit a global model — does best among supervised classifiers, consistent with this view.

### Where the leaderboard lives

* Long-form parquet on S3: [`s3://oa-bolinas/snakemake/analysis/supervised_vep/results/leaderboard.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/cf724e8e69ae8e98af0a2288549396fc96b76a1c/snakemake/analysis/supervised_vep) (910 rows: 820 supervised + 90 baseline; `score_type/subset/value/se/n_pairs/n_ties` + `model/dataset/recipe/classifier/family`).
* Local snapshot for this comment: [`scratch/iter1_leaderboard.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/cf724e8e69ae8e98af0a2288549396fc96b76a1c/scratch/iter1_leaderboard.parquet) (will commit alongside this comment).

### Caveats

1. **BFS hparam grids = one value per knob.** `C=1.0` for LogReg / LinearSVC, default sklearn KNN at `k=25`, XGBoost at `max_depth=6` + default learning rate. The right hparam is data-specific and these may be off-optimum — particularly for high-D recipes where ridge strength matters a lot. Iter-1b should refine top performers (`sym_concat × xgboost`, KNN cells) with wider grids.
2. **3 mendelian × big-feature × LinearSVC cells were skipped** — `mean_concat_plus_llr/mean_delta/sym_concat × linearsvc` ran >1h each without converging in the previous attempt. Configured via [`skip_combos` in `config.yaml`](https://github.com/Open-Athena/bolinas-dna/blob/cf724e8e69ae8e98af0a2288549396fc96b76a1c/snakemake/analysis/supervised_vep/config/config.yaml). Revisit with `dual=True` + tighter `max_iter` if LinearSVC looks competitive elsewhere.
3. **Single backbone (`exp166-p1B`), single window (255+BOS), single pool (mean over last layer).** Other backbones / multi-layer features / different pool types are deferred to iter-3+.

### Iter-1b next

Targeted refinement on the top cells:

* `mendelian × sym_concat × xgboost` — wider XGBoost grid (depth, lr, n_estimators with early stopping).
* `complex_traits × abs_delta_plus_scalars × knn` and `eqtl × sym_concat × knn` — sweep `n_neighbors` more carefully.
* Add an L1 LogReg variant: with D≈6k features and only ~5k training rows, sparsity may help where L2 doesn't.
* Add a **single-feature supervised baseline**: `embed_last_l2` alone through a scalar LogReg / sigmoid. If that recovers ≈baseline numbers, it confirms the high-D dilution hypothesis and rules out "training data is too small to learn anything."

### Resources

m7i.4xlarge (16 vCPU, us-east-2). The full BFS sweep with iter-1 code took ~1h35 across two runs (some pre-fixed already-cached jobs from a cancelled first attempt were skipped on resume). The 3 skipped LinearSVC × big-mendelian cells alone would have been another 30-60 min. Cluster `supervised-vep-cpu` still up for iter-1b.

Compute commits: [iter-1 pipeline](https://github.com/Open-Athena/bolinas-dna/commit/7895a3f) → [iter-1 fix xgboost+keep-going](https://github.com/Open-Athena/bolinas-dna/commit/6ea7df2) → [iter-1 skip slow linearsvc cells](https://github.com/Open-Athena/bolinas-dna/commit/cf724e8).

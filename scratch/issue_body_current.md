# Question

Can a small **supervised** head trained on top of `exp166-p1B`'s frozen embeddings — and later LoRA-tuned weights — materially beat the zero-shot recipes identified in #175 across the three bolinas matched-pair eval datasets, when constrained to the train split via chrom-grouped cross-validation?

#175 closed on a per-dataset best zero-shot recipe (e.g. `rank-mean(minus_llr, embed_l2_flat_last, alphagenome_max_l2)` for mendelian, with `exp166-p1B` competitive across the board). The natural next step is to ask whether learning a classifier on the same embeddings — without leaking test variants — does meaningfully better, and whether LoRA fine-tuning then adds more on top. Output is **insights** (which recipes work, by how much, and where the asymmetric ref/alt signal is or isn't useful), not a single fixed leaderboard.

# Scope

**In:**

- **Model:** `exp166-p1B` only (1B zoonomia generalist; best uniform coverage in #175 §1, §4). Native window (255 bp DNA + BOS = 256 tokens).
- **Datasets:** `bolinas-dna/{evals_mendelian_traits, evals_complex_traits, evals_eqtl}`, `split='train'` only. Test held for the final-eval pass per the #175 convention.
- **CV:** 3-fold chrom-grouped on the 12 train-split chromosomes (~4 chroms per fold, 3 fits per recipe). Out-of-fold predictions concatenated and scored through the existing `compute_pairwise_metrics` for direct comparability to #161 / #162 / #172.
- **One global classifier per dataset**; per-subset metrics sliced post-hoc from OOF predictions (TraitGym recipe).
- **Symmetry constraint:** features must be invariant under ref↔alt swap for `complex_traits` and `eqtl` (where ref/alt direction is biologically arbitrary). Mendelian gets an asymmetric probe alongside the symmetric default to quantify the value of directionality.
- **Classifiers:** sklearn LogReg, LinearSVC, KNN, XGBoost — both standard and pair-aware (rank-LogReg / `objective='rank:pairwise'`) variants. Frozen backbone, sklearn-scale fits.
- **LoRA fine-tuning** of `exp166-p1B` with a small classification head and a pair-aware ranking loss. Sequenced after frozen-embedding baselines.

**Out (deferred):**

- Test split (held for final-eval pass).
- RC averaging (#175 §2 — strong in zero-shot but adds a doubling factor to compute; revisit if frozen baselines plateau).
- Other backbones (exp55 / exp58 / exp59 / exp136 already characterized in #175).
- Larger windows (512+). Start at native; widen only if signal warrants.
- TraitGym-style AUPRC-by-chrom helper — all metrics flow through the existing `compute_pairwise_metrics` so this investigation is directly comparable to the frozen leaderboards #161 / #162 / #172.

# Approach

## Symmetry

`complex_traits` and `eqtl` have no biological ref/alt direction (effect-size sign is arbitrary in fine-mapping). Features used on those datasets must be invariant under ref↔alt swap. Mendelian has a meaningful direction (ref = wild-type, alt = variant).

The plan adopts a **symmetric default recipe consistent across all 3 datasets**, plus an **asymmetric probe** mendelian-only that quantifies the value of directionality.

## Iteration 0 — feature cache

Extract a mean-pooled-embedding cache for `exp166-p1B` × all 3 train splits, written to S3 as parquet (one row per variant).

- Reuse `biofoundation.inference` plumbing (Trainer with `data_transform_on_the_fly=True`, same forward as `evals_v2`); extend the model-side scoring helper to return mean-pooled ref/alt vectors plus the standard zero-shot scalars.
- **Pool:** mean over token positions, last hidden state only. 2 D-dim blocks per variant (`mean_ref`, `mean_alt`) + the cached scalars (`llr`, `abs_llr`, `minus_llr`, plus the flattened `embed_l2_flat_last` scalar from #175). At D ≈ 2048 fp16, ~8 KB/variant × 15,560 variants ≈ 120 MB total.
- The TraitGym `innerprod` feature `(emb_ref * emb_alt).sum(seq)` is also computed in this stage — it can't be reconstructed from mean pools because it's per-channel summed over tokens, not pooled-then-multiplied.

## Iteration 1 — frozen-embedding linear baselines

For each (dataset, feature recipe, classifier), run 3-fold chrom-grouped CV, concatenate OOF predictions, score via existing `compute_pairwise_metrics` to get per-subset PairwiseAccuracy ± SE, `_global_`, `_macro_avg_`.

**Symmetric feature recipes (all 3 datasets):**

| recipe | construction | dim |
|---|---|---|
| `abs_delta` | \|mean_alt − mean_ref\| | D |
| `prod` | mean_ref ∘ mean_alt | D |
| `sym_concat` | concat(mean_ref + mean_alt, \|mean_alt − mean_ref\|) | 2 D |
| `traitgym_innerprod` | (emb_ref ∘ emb_alt).sum(seq) | D |
| `abs_delta + abs_llr + embed_l2` | append zero-shot symmetric scalars to `abs_delta` | D + small |

**Asymmetric probe (mendelian-only):**

| recipe | construction | dim |
|---|---|---|
| `mean_delta` | concat(mean_ref, mean_alt, mean_alt − mean_ref) | 3 D |
| `mean_concat + llr` | concat(mean_ref, mean_alt) + signed LLR | 2 D + 1 |

The mendelian-only delta between best symmetric and best asymmetric recipe quantifies how much directionality info is worth.

**Classifiers** (sklearn `SimpleImputer(mean) → StandardScaler → clf`):

Hyperparameter grids are **starting points** — every fit is checked for boundary hits in the chosen value and the grid is widened if the selected hparam sits at the min or max.

- `LogisticRegression(class_weight='balanced')`, L2.
- `LinearSVC(class_weight='balanced')`. RBF SVM skipped — D too high.
- `KNeighborsClassifier`, `metric='euclidean'` on scaled features.
- `XGBClassifier` with `scale_pos_weight` and inner-fold early stopping for `n_estimators`.

**Pair-aware classifier:** PairwiseAccuracy is a ranking metric (within each `match_group`, all that matters is `score(pos) > score(neg)`). Fitting a linear classifier on **pair-difference vectors** `Δ = feat(pos) − feat(neg)` with labels ±1 directly optimizes this. At inference, score each variant as `w · feat(v)`; OOF eval unchanged. For XGBoost, the analogue is `objective='rank:pairwise'` with `match_group` as group IDs.

## Iteration 2 — LoRA fine-tuning

Thin classification head on top of `exp166-p1B`, fine-tune backbone via PEFT LoRA, **pair-aware ranking loss** (matched to PairwiseAccuracy). Same 3-fold chrom-grouped CV. One model per dataset (different label semantics).

## Iteration 3+ — open

Driven by 0–2 results:

- RC averaging if frozen baselines plateau.
- Multi-layer features (middle + last) if last alone doesn't dominate zero-shot.
- Min/max/varpos pooling if mean is leaving signal on the table.
- Per-subset classifiers for mendelian (missense-dominated).
- Multi-task classifier across all 3 datasets if data-scarce `complex_traits` hurts.
- Contrastive pre-training of an embedding adapter on the train-split match pairs.

# Current findings

### Best-so-far per dataset (PairwiseAccuracy, train-split, 3-fold chrom-grouped OOF)

Headline metric = **global PA** (single PA over all matched pairs in the train split; more statistical power than macro). Macro PA (unweighted mean of per-subset PA for subsets with n_pairs ≥ 30) reported alongside for the leaderboard-comparable view. Updated as new best recipes are found.

Deltas computed against (a) the **LLR-protocol baseline** = the leaderboard default for each dataset (`minus_llr` for mendelian, `abs_llr` for complex_traits / eqtl), and (b) the alternative zero-shot **`embed_last_l2`** (flattened ref-vs-alt L2 distance, identified in #175 §1 as competitive).

| dataset | best recipe so far | **global PA** | macro PA | Δ global vs LLR | Δ global vs `embed_last_l2` | Δ macro vs LLR |
|---|---|---:|---:|---:|---:|---:|
| `mendelian_traits` | **`logreg_l2(embed_last_l2, minus_llr)`** wide C | **0.7462** ± 0.006 | **0.7750** ± 0.018 | +0.011 vs `minus_llr` (0.7350) | +0.024 vs `embed_last_l2` (0.7225) | +0.058 vs `minus_llr` (0.7175) |
| `complex_traits` | zero-shot `embed_last_l2` (untrained) | **0.5824** ± 0.021 | **0.6541** ± 0.032 | +0.047 vs `abs_llr` (0.5355) | — (= baseline) | +0.061 vs `abs_llr` (0.5927) |
| `eqtl` | zero-shot `embed_last_l2` (untrained) | **0.5306** ± 0.010 | **0.5360** ± 0.022 | +0.008 vs `abs_llr` (0.5230) | — (= baseline) | +0.013 vs `abs_llr` (0.5225) |

Baseline numbers were computed by running each cached zero-shot scalar through `compute_pairwise_metrics` directly on the train split, so they're drop-in comparable to #161 / #162 / #172.

### Four findings driving the story

1. **Iter-1's BFS supervised cells were C-undertuned, not capacity-limited.** With wide C grids, the same pair-aware classifier on the same `sym_concat` mendelian recipe lifts from 0.524 (iter-1 BFS, C=1.0) to **0.7164** (iter-1e, C=1e-4) — a +0.193 macro PA gain from a single hyperparameter. Standard logreg with wide C on the 2-scalar mendelian recipe achieves the current champion 0.7750.
2. **High-D supervised on mean-pooled embeddings is still dominated by the zero-shot scalar.** Even with proper C tuning, the high-D recipes (D=1920+) don't reach the 2-scalar champion. A single-feature OOF on `embed_last_l2` reproduces the zero-shot number bit-identically — confirming that the extra D dims in pool-based recipes are *dilution*, not *additional signal*.
2. **Low-dim composites of zero-shot scalars do beat zero-shot on mendelian.** A pure rank-mean of two scalars (`embed_last_l2`, `minus_llr`) lifts mendelian macro PA by +0.027 — better than any iter-1 supervised cell. The win comes from across-the-board per-subset gains (missense +0.023, splicing +0.051, synonymous +0.121, …).
3. **The same composite *hurts* on complex_traits and eqtl** because `minus_llr` is near-noise on those datasets (their zero-shot LLR macros are ~0.51-0.59, close to chance). Adding it to a composite drags `embed_last_l2` down by 0.03-0.05 on the biggest subsets.

### Implications for iter-1+

- **Dataset-specific recipes are needed** (matches #175 §1 conclusion). The right scalar set is `(embed_last_l2, minus_llr)` on mendelian but just `embed_last_l2` alone on complex / eqtl.
- **The supervised investigation should narrow** from "D=1920+ pool blocks + classifier" to "small linear (or rank) combination of zero-shot scalars". The high-D direction is a feature-design dead-end at this sample size (~5K train rows per chrom-grouped outer fold, vs D ≥ 1920 features).
- **LoRA fine-tuning (iter-2) is the remaining lever** — a learned backbone could plausibly close the gap on complex / eqtl where the frozen-embedding ceiling (`embed_last_l2` alone = 0.654 / 0.536) is the binding constraint.

# Open questions / next steps

- [x] **iter-0**: feature cache built (mean-pooled ref/alt + TraitGym innerprod + zero-shot scalars) for `exp166-p1B` × 3 datasets. [`#180 comment 4444428353`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4444428353).
- [x] **iter-1**: 82 (recipe × classifier) supervised combos through chrom-grouped 3-fold OOF. **Negative result**: high-D supervised underperforms zero-shot on 2/3 datasets. [`#180 comment 4445512959`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4445512959).
- [x] **iter-1b**: single-feature OOF + low-dim rank-mean / LogReg of zero-shot scalars. **Positive result on mendelian**: `rank-mean(embed_last_l2, minus_llr) = 0.7682` beats both zero-shot and every iter-1 cell. [`#180 comment 4445555064`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4445555064).
- [x] **iter-1c/d/e**: pair-aware deep-dive + wide C grids. **Iter-1 BFS was C-undertuned**; pair-aware on high-D mendelian × `sym_concat` jumps 0.524 → 0.7164 with proper C. New mendelian champion: `logreg(embed_last_l2, minus_llr)` wide C = **0.7750**. Pair-aware ties std logreg in the low-dim regime; high-D std-logreg-with-wide-C still untested. [`#180 comment 4445776202`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4445776202).
- [x] **iter-1f**: std logreg with wide C on high-D recipes confirms pair-aware ≈ std logreg at high-D (both lift from BFS ~0.6 → ~0.72 with proper C). [`#180 comment 4445956874`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4445956874).
- [x] **iter-2a** (LoRA, no head, score = pairwise L2 distance over flat hidden state): complex_traits 3-fold OOF ties frozen baseline (0.5798 vs 0.5824 global) and eqtl single-fold sweep shows the same null pattern. [`#180 comment 4450851925`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4450851925) + wrap-up below.
- [x] **iter-2b** (LoRA + MLP head on `|mean(alt) − mean(ref)|`): all 6 head configs overfit (train_PA up, val_PA flat or down) and tie or fall below frozen on outer-test. See wrap-up.
- [x] **Iter-2 wrap-up**: no LoRA variant breaks the frozen `embed_last_l2` ceiling on complex_traits or eqtl. Mendelian intentionally skipped — zero-shot already strong there (champion `logreg(embed_last_l2, minus_llr)` = 0.7462 global / 0.7750 macro).

# Status — closed direction

Frozen-embedding ceiling on complex_traits / eqtl is the binding constraint. No supervised approach tested (linear, non-linear, frozen, LoRA-tuned) lifts global PA above `embed_last_l2` on those two datasets. On mendelian, a tiny low-dim composite of two zero-shot scalars (`embed_last_l2 + minus_llr`) is the strongest recipe found.

# Tracking

Description = current state. Comments = append-only iteration log with commit-pinned permalinks. Pipeline README (under `snakemake/analysis/supervised_vep/`, once created) = how-to-run only.

Reference: closed predecessor #175.

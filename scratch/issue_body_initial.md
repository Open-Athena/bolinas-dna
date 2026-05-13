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

_None yet — iter-0 (feature cache) in progress._

# Open questions / next steps

- [ ] **iter-0**: extend `biofoundation.model.scoring` with a `compute_llr_and_pooled_embeddings` helper returning mean-pooled ref/alt vectors + TraitGym innerprod + the existing zero-shot scalars.
- [ ] **iter-0**: scaffold `snakemake/analysis/supervised_vep/` (copy `evals_v2` structure) and a `src/bolinas/supervised/` library module.
- [ ] **iter-0**: run feature cache on SkyPilot for `exp166-p1B` × {`evals_mendelian_traits`, `evals_complex_traits`, `evals_eqtl`} train splits; sanity-check that `minus_llr` / `abs_llr` from the cache reproduces #161 / #162 / #172 PairwiseAccuracy within MC noise.
- [ ] **iter-1**: chrom-grouped 3-fold CV splitter + OOF aggregator + symmetric / asymmetric feature builders + sklearn classifier wrappers + pair-aware variants. Run all (dataset, recipe, classifier) combos.
- [ ] **iter-2**: LoRA fine-tuning with pair-aware ranking loss.

# Tracking

Description = current state. Comments = append-only iteration log with commit-pinned permalinks. Pipeline README (under `snakemake/analysis/supervised_vep/`, once created) = how-to-run only.

Reference: closed predecessor #175.

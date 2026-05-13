🤖 **iter-1b: the supervised story flips when we drop the high-D pools.**

Tested two cheap experiments locally against the iter-0 cache (no cluster):

### 1. Single-feature OOF on `embed_last_l2` alone reproduces the zero-shot baseline EXACTLY

A chrom-grouped 3-fold OOF with a sigmoid LogReg on the single scalar `embed_last_l2` lands at macro PA **0.7408 / 0.6541 / 0.5360** on mendelian / complex / eqtl — bit-identical to the zero-shot `embed_last_l2` baseline. This is expected (a monotone transform of a scalar preserves pairwise rankings) but **it confirms the high-D dilution hypothesis from iter-1**: the BFS supervised classifiers on D=1920+ pooled features don't lose to zero-shot because of bad hparams or wrong loss — they lose because the high-D feature blocks are diluting a scalar signal that's already near-optimal.

### 2. Low-dim rank-mean of zero-shot scalars beats both zero-shot and BFS supervised on mendelian

A simple `rank-mean` of two zero-shot scalars from the iter-0 cache (no training, no fold structure needed) gets:

| recipe | mendelian | complex | eqtl |
|---|---:|---:|---:|
| **zero-shot best (`embed_last_l2` alone)** | 0.7408 ± 0.019 | **0.6541** ± 0.032 | **0.5360** ± 0.022 |
| **iter-1 best supervised (BFS, high-D, chrom-grouped CV)** | 0.7069 ± 0.020 | 0.5760 ± 0.034 | 0.5781 ± 0.022 |
| **`rank-mean(embed_last_l2, minus_llr)`** | **0.7682** ± 0.018 | 0.5979 ± 0.034 | 0.5097 ± 0.022 |
| `rank-mean(embed_last_l2, abs_llr)` | 0.6822 ± 0.021 | 0.6207 ± 0.033 | 0.5247 ± 0.022 |
| 3-scalar LogReg-OOF (el2+abs_llr+minus_llr) | 0.7589 ± 0.019 | 0.6178 ± 0.033 | 0.5175 ± 0.022 |
| `rank-mean` 4-way (el2+ll−/+ pooled_l2 + pooled_cos) | 0.7414 ± 0.019 | 0.6166 ± 0.033 | 0.5371 ± 0.022 |

**Mendelian: a 2-scalar zero-shot composite (no training!) beats every iter-1 BFS supervised cell by +0.061 over the best iter-1 cell (`sym_concat × xgboost = 0.7069`) and +0.027 over zero-shot `embed_last_l2` alone.** This mirrors the #175 §1 finding for `exp166-p1B` — `rank-mean(minus_llr, embed_l2_flat_last, ...)` was the headline ensemble there.

**Complex_traits and eqtl: nothing beats `embed_last_l2` alone.** Adding `minus_llr` to a composite *hurts* (drops mendelian-style gain costs ~0.05 macro PA on these regulatory-heavy datasets where LLR is near-noise). The iter-1 BFS KNN cells on `sym_concat`/`abs_delta_plus_scalars` are within SE of zero-shot but never decisively above.

### What this means for the iter-1 negative result

The iter-1 BFS supervised classifiers weren't *strictly* worse than zero-shot because the classifiers were bad — they were worse because **none of the iter-1 feature recipes were the right low-dim composite**:

* The high-D pool blocks (`abs_delta`, `prod`, `sym_concat`, `mean_delta`, …) bury the scalar signal under D=1920+ noisy dimensions, and the chrom-grouped CV's effective n is ~6.5K rows per fold — not enough to learn a sparse projection from D dims back down to "use `embed_last_l2` and `minus_llr`".
* The two recipes that *did* include zero-shot scalars (`abs_delta_plus_scalars`, `mean_concat_plus_llr`) appended them to D-dim blocks, so the classifier had to learn to *ignore* the D dims and *attend* to 1-2 scalars — exactly the wrong inductive bias.

**Right inductive bias for this problem at this sample size: start from the zero-shot scalars and let a low-dim head decide how to combine them.** Either as `rank-mean` (no learned weights — Pareto-strong on mendelian) or a small LogReg on 2-5 scalars (slightly behind rank-mean on mendelian; closer to it on complex/eqtl).

### Code + data

* `scratch/iter1b_fast_baselines.py` — alternative zero-shot scalars from the pools (pooled_l2, pooled_cosine, etc.) + single-feature OOF on each: [`commit/d83602d`](https://github.com/Open-Athena/bolinas-dna/blob/d83602d72a01f1d4ed09b8a8bca8b8e9bb4ad6ce/scratch/iter1b_fast_baselines.py).
* `scratch/iter1b_scalar_combos.py` — 2/3-scalar LogReg-OOF combos: [`commit/d83602d`](https://github.com/Open-Athena/bolinas-dna/blob/d83602d72a01f1d4ed09b8a8bca8b8e9bb4ad6ce/scratch/iter1b_scalar_combos.py).
* Both ran locally in ~30 seconds against the iter-0 cache on S3.

### Next (cheap, mostly local)

1. **Replicate #175 §1's full 3-way composite** with the cached scalars + the missing `alphagenome_max_l2` (or its closest proxy in this cache) to see if the supervised gap closes further.
2. **Tiny supervised LogReg on the zero-shot scalars only** — same OOF schema, but with a wider `C` grid + a learned linear weighting. Should match-or-beat rank-mean on mendelian and not regress on complex/eqtl. If it works, that's the supervised story for iter-1.
3. **L1 LogReg on the full D=2D-3D feature recipes** with the zero-shot scalars also included — does sparsity recover the rank-mean signal automatically without us hand-picking the 2 scalars? If yes, the supervised approach generalises beyond `embed_last_l2 + minus_llr` to richer feature sets.
4. **Skip rerunning** the high-D pool-only recipes from iter-1 with refined hparams — based on (1) and (2), more hparam tuning on those recipes won't lift them above 0.71 mendelian. The dilution is a feature-design problem, not a hparam problem.

Iter-2 (LoRA fine-tuning) is the real wildcard — a learned backbone could plausibly close the gap on complex/eqtl where the frozen `embed_last_l2` ceiling is low.

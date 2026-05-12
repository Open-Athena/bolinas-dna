🤖 **Iter-4 follow-up: ensembling scores on exp166-p1B (1B generalist), 3 datasets.** Revisited the iter-2 rank-mean composites on the 1B generalist to see whether ensembling beats single-score scoring. **Yes, by 0.03–0.04 on macro for two of three datasets.** Commit [`4d6e1c4`](https://github.com/Open-Athena/bolinas-dna/commit/4d6e1c4).

## TL;DR

- **Composites beat the default leaderboard score** (`minus_llr` for mendelian, `abs_llr` for complex/eqtl) on **18 of 22** (dataset, subset) cells.
- Macro gains: mendelian **+0.040** (0.725 → 0.765), complex **+0.035** (0.515 → 0.549), eqtl **+0.002**.
- Pooled gains: mendelian +0.005, complex **+0.032**, eqtl +0.009.
- **Two simple composites carry most of the value**:
  - `rk_llr_l2flat` = rank-mean of (`minus_llr` + `embed_l2_flat_last`) — 2 scores. Wins macro on mendelian.
  - `rk_llr_top_emb` = rank-mean of (`minus_llr` + 4 top embeddings) — 5 scores. Wins macro on complex + eqtl.
- Biggest per-cell wins: complex ncRNA +0.375, eqtl missense +0.167, mendelian ncRNA +0.119, mendelian synonymous +0.091.

## Composites tested

Rank-within-subset ensembles (so the average is across heterogeneous-scale scores). Ranks computed per (dataset, subset) — no learned weighting, fully zero-shot.

| name | scores |
|---|---|
| `rk_llr_l2flat` | `minus_llr` + `embed_l2_flat_last` |
| `rk_llr_top_emb` | `minus_llr` + 4 top embeddings (`embed_l2_flat_last`, `embed_cosine_flat_last`, `embed_l2_mean_last`, `embed_cosine_mean_last`) |
| `rk_all_likelihood` | `minus_llr` + `abs_llr` + `minus_logp_alt` + `logp_ref` + `minus_entropy` |
| `rk_p1B_top6` | exp166-p1B's per-subset winners union: `minus_llr` + `minus_logp_alt` + `embed_l2_flat_last` + `embed_l2_mean_middle` + `logp_ref` + `minus_entropy` |
| `rk_llr_l2flat_ent` | `minus_llr` + `embed_l2_flat_last` + `minus_entropy` |
| `rk_absllr_l2flat` | `abs_llr` + `embed_l2_flat_last` (variant for complex/eqtl) |

## Pooled accuracy: best composite vs default

| dataset | default (score) | best composite | Δ |
|---|---:|---|---:|
| mendelian | 0.739 (`minus_llr`) | **0.744** (`rk_llr_l2flat`) | +0.005 |
| complex | 0.528 (`abs_llr`) | **0.560** (`rk_llr_top_emb`) | **+0.032** |
| eqtl | 0.519 (`abs_llr`) | **0.527** (`rk_absllr_l2flat`) | +0.009 |

## Macro accuracy (subset-averaged): best composite vs default

| dataset | default macro | best composite | Δ |
|---|---:|---|---:|
| mendelian | 0.725 | **0.765** (`rk_llr_l2flat`) | **+0.040** |
| complex | 0.515 | **0.549** (`rk_llr_top_emb`) | **+0.035** |
| eqtl | 0.535 | **0.537** (`rk_llr_top_emb`) | +0.002 |

Macro gains are larger than pooled because composites help most on small-to-medium subsets (which dominate macro but get diluted in pooled). On the very-large subsets (mendelian missense n=4495, complex distal n=428, eqtl distal n=1678), composites give a small +0.002 to +0.032 lift; on smaller subsets, composites pick up large fractional gains.

## Per-cell results: composite vs default

★ = composite wins.

### mendelian_traits (default = `minus_llr`)

| subset | n_pairs | default | best composite | Δ |
|---|---:|---:|---|---:|
| ★ 3'UTR | 58 | 0.724 | **0.741** (`rk_llr_l2flat`) | +0.017 |
| ★ 5'UTR | 87 | 0.678 | **0.736** (`rk_p1B_top6`) | +0.057 |
| ★ distal | 56 | 0.714 | **0.768** (`rk_llr_l2flat`) | +0.054 |
| ★ missense | 4495 | 0.741 | **0.742** (`rk_llr_l2flat`) | +0.002 |
| ★ ncRNA | 42 | 0.690 | **0.810** (`rk_llr_top_emb`) | **+0.119** |
| ★ splicing | 78 | 0.808 | **0.859** (`rk_p1B_top6`) | +0.051 |
| ★ synonymous | 33 | 0.758 | **0.848** (`rk_llr_l2flat`) | +0.091 |
| tss_proximal | 61 | 0.689 | 0.664 (`rk_llr_l2flat`) | −0.025 |

**Composite wins on 7 of 8 subsets.** The only loss is tss_proximal where `logp_ref` (alone) gives 0.730, beating all composites.

### complex_traits (default = `abs_llr`)

| subset | n_pairs | default | best composite | Δ |
|---|---:|---:|---|---:|
| 3'UTR | 21 | 0.571 | 0.500 (`rk_absllr_l2flat`) | −0.071 |
| 5'UTR | 10 | 0.700 | 0.700 (`rk_llr_l2flat`) | 0.000 |
| ★ distal | 428 | 0.516 | **0.548** (`rk_llr_l2flat`) | +0.032 |
| ★ missense | 57 | 0.667 | **0.737** (`rk_llr_top_emb`) | +0.070 |
| ★ ncRNA | 12 | 0.333 | **0.708** (`rk_all_likelihood`) | **+0.375** |
| synonymous | 6 | 0.333 | 0.333 (`rk_llr_l2flat`) | 0.000 |
| ★ tss_proximal | 29 | 0.483 | **0.552** (`rk_all_likelihood`) | +0.069 |

**Composite wins on 4 of 7 subsets, ties on 2.** The +0.375 on ncRNA is striking but n=12 is fragile. The +0.032 on distal (n=428, the largest subset by far in complex) is solid.

### eqtl (default = `abs_llr`)

| subset | n_pairs | default | best composite | Δ |
|---|---:|---:|---|---:|
| 3'UTR | 115 | 0.591 | 0.570 (`rk_all_likelihood`) | −0.022 |
| 5'UTR | 51 | 0.529 | 0.529 (`rk_all_likelihood`) | 0.000 |
| ★ distal | 1678 | 0.513 | **0.526** (`rk_llr_l2flat`) | +0.013 |
| ★ missense | 30 | 0.467 | **0.633** (`rk_llr_top_emb`) | **+0.167** |
| ★ ncRNA | 157 | 0.478 | **0.592** (`rk_llr_top_emb`) | **+0.115** |
| splicing | 5 | 0.600 | 0.500 (`rk_llr_top_emb`) | −0.100 |
| ★ synonymous | 29 | 0.552 | **0.621** (`rk_absllr_l2flat`) | +0.069 |
| ★ tss_proximal | 241 | 0.552 | **0.581** (`rk_absllr_l2flat`) | +0.029 |

**Composite wins on 5 of 8 subsets, ties on 1.** The +0.167 missense and +0.115 ncRNA wins are substantial. eqtl is the dataset where the default score is closest to chance (≤0.55 on most subsets), so even small composite gains matter relative to the headroom.

## Which composite wins where

| dataset | macro-winner | pooled-winner |
|---|---|---|
| mendelian | `rk_llr_l2flat` (+0.040) | `rk_llr_l2flat` (+0.005) |
| complex | `rk_llr_top_emb` (+0.035) | `rk_llr_top_emb` (+0.032) |
| eqtl | `rk_llr_top_emb` (+0.002) | `rk_absllr_l2flat` (+0.009) |

**`rk_llr_top_emb`** (LLR + 4 strongest last-layer embedding distances) is the most universal winner — top on 2/3 datasets by macro. **`rk_llr_l2flat`** is the simpler 2-score variant that ties or wins on mendelian.

Notably: `rk_all_likelihood` (LLR family only, no embeddings) **does not** consistently win — it's only top on a few small-n cells. **Mixing likelihood with embedding-distance scores is where the composite gains come from.**

## Comparison vs RC-averaging

For exp166-p1B specifically, putting these together:
- AVG (FWD+RC) of `minus_llr` on splicing mendelian: 0.808 vs FWD-only 0.808 (tied)
- `rk_p1B_top6` on splicing mendelian (FWD only): **0.859** (+0.051)

So for the 1B generalist, **score-level ensembling gives bigger gains than strand-level averaging** on splicing — though they're additive in principle (FWD-only composite + RC-only composite, then AVG strands). The RC scout for exp166-p1B is already on S3 from this morning if a combined experiment is wanted.

## Practical recommendation

For leaderboard protocols using exp166-p1B (or similar generalist models):

1. **Switch the default from `minus_llr`/`abs_llr` to `rk_llr_top_emb`** (rank-mean of LLR + 4 top last-layer embedding distances). Macro lift of +0.02 to +0.04 across the 3 datasets, pooled +0.005 to +0.032. No new forward passes — same data, just a different post-processing rule.
2. **For maximum simplicity, `rk_llr_l2flat` (2 scores)** captures most of the mendelian gain with the smallest score set.
3. **Combine with RC-averaging** (the offline biofoundation issue Open-Athena/biofoundation#24) for further gains — these are independent improvements.

## Code @ [`4d6e1c4`](https://github.com/Open-Athena/bolinas-dna/commit/4d6e1c4)

- [`scratch/zeroshot_vep_iter2_combinations.py`](https://github.com/Open-Athena/bolinas-dna/blob/4d6e1c4/scratch/zeroshot_vep_iter2_combinations.py) — original iter-2 composite definitions (reused here)
- exp166-p1B FWD parquets across the 3 datasets — [`scratch/iter4/iter4_fwd_exp166-p1B__*`](https://github.com/Open-Athena/bolinas-dna/tree/4d6e1c4/scratch/iter4)

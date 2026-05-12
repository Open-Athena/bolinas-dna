🤖 **Iter-4 follow-up: simple 2-way of `embed_l2_flat_last` + AlphaGenome (vs LLR + AG).** You asked whether replacing the LLR side of the 2-way with an embedding-distance score gives a different/better answer. Result: **yes for complex_traits** — embed+AG beats both LLR+AG and AG-alone there. Commit [`69e92cc`](https://github.com/Open-Athena/bolinas-dna/commit/69e92cc).

## TL;DR

- **mendelian**: LLR+AG still wins macro (0.764), embed+AG is close (0.754, with `embed_l2_flat_last`).
- **complex_traits**: **embed+AG wins** — macro 0.618 with `embed_l2_flat_last`, beating both AG-alone (0.607) and the prior LLR+AG (0.583). `embed_minus_dot_mean_last` + AG goes higher still at 0.627.
- **eqtl**: AG-alone is still best (0.706 macro); no embed-pair beats it (best embed-pair is `embed_l2_mean_last` + AG at 0.672, still −0.034 vs AG-alone).

## Headline numbers (macro)

Using `embed_l2_flat_last` as the embedding side (the most general-purpose embedding score across iter-4):

| dataset | embed alone | AG alone | LLR + AG | embed + AG |
|---|---:|---:|---:|---:|
| mendelian | 0.741 | 0.700 | **0.764** | 0.754 |
| complex | 0.548 | 0.607 | 0.583 | **0.618** |
| eqtl | 0.490 | **0.706** | 0.631 | 0.614 |

Same comparison, pooled (whole-dataset PA, no subset weighting):

| dataset | embed alone | AG alone | LLR + AG | embed + AG |
|---|---:|---:|---:|---:|
| mendelian | 0.722 | 0.543 | 0.708 | 0.694 |
| complex | 0.578 | **0.621** | 0.595 | **0.627** |
| eqtl | 0.531 | **0.621** | 0.580 | 0.583 |

On **complex pooled**, embed+AG (0.627) edges out AG-alone (0.621) by +0.006 — small but the only ensemble that beats AG-alone pooled on complex. Combined with the macro lift of +0.011, this is the cleanest "embed+AG > AG-alone" case.

## Across embedding candidates

To check whether `embed_l2_flat_last` is the right choice or if another embed pool wins, scanned 7 candidates (macro):

**complex_traits** macro for `<embed> + AG` (vs AG-alone 0.607):

| embed score | embed alone | 2-way | Δ vs AG |
|---|---:|---:|---:|
| `embed_minus_dot_mean_last` | 0.539 | **0.627** | **+0.020** |
| `embed_l2_flat_last` | 0.548 | 0.618 | +0.011 |
| `embed_l2_mean_last` | 0.522 | 0.577 | −0.030 |
| `embed_cosine_mean_last` | 0.525 | 0.567 | −0.040 |
| `embed_l2_flat_middle` | 0.490 | 0.564 | −0.043 |
| `embed_cosine_flat_last` | 0.554 | 0.556 | −0.051 |
| `embed_minus_dot_flat_last` | 0.388 | 0.539 | −0.068 |

The `embed_minus_dot_mean_last` winner is interesting — its embed-alone score (0.539) is *worse* than `embed_l2_flat_last` (0.548), but it combines better with AG. Different feature, more independent from what AG captures. (This kind of "ensemble-friendly embedding" picking would normally need a hold-out set to avoid over-fitting; here it's just exploratory.)

**eqtl** macro for `<embed> + AG`: best is `embed_l2_mean_last` + AG at 0.672, still worse than AG-alone 0.706. **No 2-way beats AG on eqtl.**

**mendelian** macro: top embed-pair is `embed_l2_flat_last` + AG at 0.754, vs LLR+AG at 0.764 — LLR side stays better here.

## Combined picture across all 2-way ensembles tested so far

For each dataset, best macro-accuracy 2-way ensemble:

| dataset | best 2-way | macro | Δ vs AG-alone | Δ vs LLR-alone |
|---|---|---:|---:|---:|
| mendelian | LLR + AG | 0.764 | **+0.064** | **+0.039** |
| complex | `embed_minus_dot_mean_last` + AG | 0.627 | +0.020 | +0.112 |
| eqtl | (none — AG-alone wins) | 0.706 | 0 | +0.171 |

## Why does the embed side help on complex but not eqtl?

Hypothesis: complex_traits has more diverse variant types (distal n=428 dominates but with non-trivial coding-adjacent subsets), so the embedding's "global representation shift" picks up complementary signal to AG's regulatory readout. eqtl is much more uniformly regulatory (distal n=1678 dominates a regulatory-heavy distribution), and AG was designed for exactly that — so the embedding adds noise. The complex/eqtl distinction in the bolinas-dna eval datasets isn't just sample size; it's variant composition.

## Updated practical recommendation

| dataset | best 2-way recipe | macro lift |
|---|---|---:|
| mendelian | `rank-mean(minus_llr, alphagenome_max_l2)` | +0.039 vs LLR, +0.064 vs AG |
| complex | `rank-mean(embed_l2_flat_last, alphagenome_max_l2)` | +0.011 vs AG, +0.103 vs LLR |
| eqtl | `alphagenome_max_l2` alone | (AG dominant; no 2-way helps) |

For a "single recipe across all 3 datasets" — there isn't one. The cleanest defaults are dataset-specific 2-ways. If forced to pick one rule, **AG alone** is the safest (always at least 0.607 macro across the 3 datasets), but you give up the +0.039 mendelian lift.

## Code

Same parquets as the prior 2-way AG comment; new analysis on top of:
- exp166-p1B FWD scores ([`scratch/iter4/iter4_fwd_exp166-p1B__win255__*.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/69e92cc/scratch/iter4))
- AlphaGenome `alphagenome_max_l2` ([`scratch/iter4/alphagenome/*.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/69e92cc/scratch/iter4/alphagenome))

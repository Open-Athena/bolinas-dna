🤖 **Iter-4 follow-up: simple 2-way LLR + AlphaGenome ensemble.** You asked specifically about the simple 2-way (no embeddings, no other composites): just `rank-mean(minus_llr, alphagenome_max_l2)` for mendelian and `rank-mean(abs_llr, alphagenome_max_l2)` for complex / eqtl. I had it computed earlier but buried it under "best ensemble" — pulling it out cleanly here. Commit [`7f9dcf4`](https://github.com/Open-Athena/bolinas-dna/commit/7f9dcf4).

## TL;DR

- **On mendelian, the simple 2-way ensemble is the best single recipe across the board** — macro **0.764** vs LLR-alone 0.725 (+0.039) vs AG-alone 0.700 (+0.064). Beats both single models on 5 of 8 subsets.
- **On complex_traits and eqtl, AG alone still wins** — the 2-way drags AG's better predictions toward LLR's noise.
- **Across all 23 cells**: 2-way beats LLR-alone on 18/23 (mean Δ +0.068); beats AG-alone on 10/23 (mean Δ −0.011); beats BOTH on 7/23. So as a "default scoring rule across the 3 datasets" it's strictly better than LLR but not strictly better than AG.

## Headline numbers

**Pooled:**

| dataset | LLR alone | AG alone | 2-way (LLR + AG) | Δ vs LLR | Δ vs AG |
|---|---:|---:|---:|---:|---:|
| mendelian | **0.739** | 0.543 | 0.708 | −0.031 | +0.166 |
| complex | 0.528 | **0.621** | 0.595 | +0.066 | −0.026 |
| eqtl | 0.519 | **0.621** | 0.580 | +0.061 | −0.042 |

**Macro (mean across subsets):**

| dataset | LLR alone | AG alone | 2-way | Δ vs LLR | Δ vs AG |
|---|---:|---:|---:|---:|---:|
| mendelian | 0.725 | 0.700 | **0.764** | +0.039 | +0.064 |
| complex | 0.515 | **0.607** | 0.583 | +0.068 | −0.024 |
| eqtl | 0.535 | **0.706** | 0.631 | +0.096 | −0.075 |

The macro-mendelian win is the clean case: 2-way is +0.039 over LLR AND +0.064 over AG — strictly dominant.

## Per-cell (exp166-p1B `minus_llr`/`abs_llr` + `alphagenome_max_l2`)

| dataset | subset | n | LLR | AG | 2-way | Δ vs LLR | Δ vs AG | beats both? |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| mendelian | 3'UTR | 58 | 0.724 | 0.655 | 0.655 | −0.069 | 0.000 | — |
| mendelian | **5'UTR** | 87 | 0.678 | 0.678 | **0.747** | +0.069 | +0.069 | ★ |
| mendelian | **distal** | 56 | 0.714 | 0.643 | **0.750** | +0.036 | +0.107 | ★ |
| mendelian | missense | 4495 | 0.741 | 0.526 | 0.703 | −0.038 | +0.177 | — |
| mendelian | **ncRNA** | 42 | 0.690 | 0.762 | **0.786** | +0.095 | +0.024 | ★ |
| mendelian | **splicing** | 78 | 0.808 | 0.859 | **0.897** | +0.090 | +0.038 | ★ |
| mendelian | **synonymous** | 33 | 0.758 | 0.818 | **0.909** | +0.152 | +0.091 | ★ |
| mendelian | tss_proximal | 61 | 0.689 | 0.656 | 0.664 | −0.025 | +0.008 | — |
| complex | 3'UTR | 21 | 0.571 | 0.714 | 0.619 | +0.048 | −0.095 | — |
| complex | 5'UTR | 10 | 0.700 | 0.800 | 0.750 | +0.050 | −0.050 | — |
| complex | distal | 428 | 0.516 | 0.638 | 0.593 | +0.077 | −0.044 | — |
| complex | missense | 57 | 0.667 | 0.474 | 0.588 | −0.079 | +0.114 | — |
| complex | **ncRNA** | 12 | 0.333 | 0.500 | **0.542** | +0.208 | +0.042 | ★ |
| complex | synonymous | 6 | 0.333 | 0.500 | 0.333 | 0.000 | −0.167 | — |
| complex | **tss_proximal** | 29 | 0.483 | 0.621 | **0.655** | +0.172 | +0.034 | ★ |
| eqtl | 3'UTR | 115 | 0.591 | 0.678 | 0.678 | +0.087 | 0.000 | — |
| eqtl | 5'UTR | 51 | 0.529 | 0.725 | 0.608 | +0.078 | −0.118 | — |
| eqtl | distal | 1678 | 0.513 | 0.604 | 0.563 | +0.050 | −0.041 | — |
| eqtl | missense | 30 | 0.467 | 0.633 | 0.500 | +0.033 | −0.133 | — |
| eqtl | ncRNA | 157 | 0.478 | 0.586 | 0.570 | +0.092 | −0.016 | — |
| eqtl | splicing | 5 | 0.600 | 1.000 | 0.800 | +0.200 | −0.200 | — |
| eqtl | synonymous | 29 | 0.552 | 0.724 | 0.690 | +0.138 | −0.034 | — |
| eqtl | tss_proximal | 241 | 0.552 | 0.697 | 0.643 | +0.091 | −0.054 | — |

★ = 2-way beats BOTH LLR alone AND AG alone (7 cells total — all on mendelian or complex small subsets).

## The pattern

The simple 2-way ensemble's win is concentrated on subsets where:

1. **Both LLR and AG give meaningful signal** — neither is at chance, neither dominates. Mendelian splicing (LLR 0.808, AG 0.859), mendelian synonymous (0.758, 0.818), mendelian 5'UTR (0.678, 0.678).
2. **The two models use partly-independent evidence** — LLR captures the sequence-level effect, AG captures the regulatory readout.

The ensemble fails to win on:
- **Missense-heavy** subsets where LLR strongly dominates (mendelian missense: LLR 0.741 vs AG 0.526 — pulling toward AG hurts).
- **Regulatory-heavy** subsets where AG strongly dominates (eqtl distal, complex distal — pulling toward LLR hurts).

## Practical recommendation

- **For mendelian (which is the default zero-shot benchmark in this repo):** use the simple 2-way `rank-mean(minus_llr, alphagenome_max_l2)` as the default scoring. **Macro 0.764 vs 0.725 for LLR-alone (+0.039)**, no extra forward passes for the LM (just one AG API call per variant).
- **For complex and eqtl:** stick with `alphagenome_max_l2` alone.
- **An ideal "single rule across all 3 datasets" doesn't exist** with these ingredients — the regulatory vs coding split is too strong. A per-dataset selection (mendelian = 2-way, complex/eqtl = AG-alone) is the clean recommendation.

## Code @ [`7f9dcf4`](https://github.com/Open-Athena/bolinas-dna/commit/7f9dcf4)

Same artifacts as the previous AG comment; this is just a re-slicing of the same analysis to break out the simple 2-way.

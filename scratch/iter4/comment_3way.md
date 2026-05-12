🤖 **Iter-4 wrap: 3-way rank-mean ensemble** — `rank-mean(default_LLR, embed_l2_flat_last, alphagenome_max_l2)` where default_LLR = `minus_llr` for mendelian and `abs_llr` for complex/eqtl. Commit [`d0c0404`](https://github.com/Open-Athena/bolinas-dna/commit/d0c0404).

## TL;DR

- **mendelian**: 3-way wins macro = **0.782**, beating LLR+AG (0.764), emb+AG (0.754), AG-alone (0.700). Clean dominant win across all 2-way alternatives.
- **complex**: 3-way macro = 0.536, **worse than emb+AG** (0.618). Adding `abs_llr` (which performs at 0.515 macro on complex) drags the ensemble down.
- **eqtl**: 3-way macro = 0.640, worse than AG-alone (0.706). AG is so dominant on eqtl that any LLM addition dilutes it.

So **no universal 3-way recipe** — the per-dataset breakdown is still the right policy.

## Macro comparison across all single + 2-way + 3-way

| dataset | LLR | emb | AG | LLR+AG | emb+AG | **3-way** | best |
|---|---:|---:|---:|---:|---:|---:|---|
| mendelian | 0.725 | 0.741 | 0.700 | 0.764 | 0.754 | **0.782** | **3-way** |
| complex | 0.515 | 0.548 | 0.607 | 0.583 | **0.618** | 0.536 | emb+AG |
| eqtl | 0.535 | 0.490 | **0.706** | 0.631 | 0.614 | 0.640 | AG-alone |

Pooled (whole-dataset PA):

| dataset | LLR | emb | AG | LLR+AG | emb+AG | 3-way |
|---|---:|---:|---:|---:|---:|---:|
| mendelian | **0.739** | 0.722 | 0.543 | 0.708 | 0.694 | 0.734 |
| complex | 0.528 | 0.578 | 0.621 | 0.595 | **0.627** | 0.604 |
| eqtl | 0.519 | 0.531 | **0.621** | 0.580 | 0.583 | 0.583 |

On pooled, the 3-way is competitive but never wins — `minus_llr` alone wins mendelian pooled (because missense n=4495 dominates), AG-alone wins eqtl pooled.

## Final per-dataset recipe (best macro)

| dataset | recipe | macro | vs default LLR |
|---|---|---:|---:|
| mendelian | `rank-mean(minus_llr, embed_l2_flat_last, alphagenome_max_l2)` (3-way) | 0.782 | +0.057 |
| complex | `rank-mean(embed_l2_flat_last, alphagenome_max_l2)` (emb+AG 2-way) | 0.618 | +0.103 |
| eqtl | `alphagenome_max_l2` alone | 0.706 | +0.171 |

## Closing

Iter-4 wraps here. Issue body will be updated with the consolidated conclusions.

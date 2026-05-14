🤖 **Iter-6 — nucleotide-dependency ("nuc-dep") downstream-effect scores on exp166-p1B, FWD+RC AVG.**

**TL;DR — nuc-dep behaves like a logit-space version of `embed_l2_flat_last`, and the user's hypothesis holds:**

- On regulatory datasets (**complex_traits**, **eqtl**), where LLR has little signal because alleles aren't under strong selection, nuc-dep matches or beats `embed_l2_flat_last`: complex Global PA 0.576 vs 0.567 (nuc-dep wins by Δ+0.009); eqtl Global PA 0.531 vs 0.523 (+0.009). These are descriptive though — paired McNemar q's are 0.41–0.69 (not significant; n_pairs constrains power).
- On **mendelian** (coding-heavy, strong selection), LLR dominates and nuc-dep underperforms both LLR (Δ=-0.019, q=8e-4 ★) and L2 (Δ=-0.010, q=0.008 ★). When the variant directly disrupts the protein-coding signal, the next-token LLR at the variant position is already the right thing.
- **Spearman ρ(nuc-dep, L2) is 0.80–0.90 within subset across all 3 datasets — much higher than ρ(nuc-dep, LLR) of 0.38–0.61, and much higher than ρ(LLR, L2) of 0.25–0.60.** Nuc-dep is structurally similar to L2 in what it captures.
- **All in logit space** — no embedding extraction needed. Computationally cheaper for downstream uses (online training-time eval, distillation, etc.).

## Setup

Per variant, 2 forward passes (REF-context, ALT-context). At each output position `i ∈ [tok_var_pos, T−1]` (positions whose AR conditioning includes the variant), compute a divergence between the renormalized 4-nucleotide softmax under REF and ALT, then aggregate over positions. 4 metrics (`jsd`/`l1`/`l2`/`linf`) × 2 aggregations (`mean`/`max`) = 8 scores. AVG over FWD + RC strands. Commit [`0444c0c`](https://github.com/Open-Athena/bolinas-dna/commit/0444c0c).

**FWD captures the genomic-downstream half of the variant's effect footprint; RC captures the genomic-upstream half (because the AR mask runs in token order, which is reversed under RC). AVG combines both → bidirectional nuc-dep, despite the unidirectional AR model.**

### mendelian_traits — Global / Macro Avg (k=8 subsets, n≥30)

| score | Global | Macro Avg |
|---|---:|---:|
| `minus_llr` (baseline) | 0.7513 ± 0.0062 | 0.7203 ± 0.0198 |
| `embed_l2_flat_last` (baseline) | 0.7424 ± 0.0062 | 0.7461 ± 0.0193 |
| `down_jsd_mean` | 0.7320 ± 0.0063 | 0.7281 ± 0.0194 |
| `down_l1_max` | 0.7173 ± 0.0064 | 0.6782 ± 0.0207 |
| `down_linf_max` | 0.7167 ± 0.0064 | 0.6813 ± 0.0208 |
| `down_jsd_max` | 0.7165 ± 0.0064 | 0.6606 ± 0.0210 |
| `down_l2_mean` | 0.7163 ± 0.0064 | 0.7309 ± 0.0197 |
| `down_linf_mean` | 0.7157 ± 0.0064 | 0.7286 ± 0.0197 |
| `down_l2_max` | 0.7151 ± 0.0064 | 0.6672 ± 0.0210 |
| `down_l1_mean` | 0.7138 ± 0.0065 | 0.7253 ± 0.0199 |

### complex_traits — Global / Macro Avg (k=2 subsets, n≥30)

| score | Global | Macro Avg |
|---|---:|---:|
| `abs_llr` (baseline) | 0.5372 ± 0.0210 | 0.6085 ± 0.0331 |
| `embed_l2_flat_last` (baseline) | 0.5674 ± 0.0209 | 0.6289 ± 0.0326 |
| `down_linf_mean` | 0.5762 ± 0.0208 | 0.6236 ± 0.0330 |
| `down_l2_mean` | 0.5745 ± 0.0208 | 0.6236 ± 0.0330 |
| `down_l1_mean` | 0.5727 ± 0.0208 | 0.6149 ± 0.0334 |
| `down_jsd_mean` | 0.5709 ± 0.0208 | 0.6430 ± 0.0315 |
| `down_l1_max` | 0.5638 ± 0.0209 | 0.6307 ± 0.0321 |
| `down_linf_max` | 0.5603 ± 0.0209 | 0.6348 ± 0.0316 |
| `down_jsd_max` | 0.5514 ± 0.0209 | 0.6149 ± 0.0326 |
| `down_l2_max` | 0.5514 ± 0.0209 | 0.6126 ± 0.0326 |

### eqtl — Global / Macro Avg (k=6 subsets, n≥30)

| score | Global | Macro Avg |
|---|---:|---:|
| `abs_llr` (baseline) | 0.5106 ± 0.0104 | 0.5072 ± 0.0224 |
| `embed_l2_flat_last` (baseline) | 0.5225 ± 0.0104 | 0.5255 ± 0.0224 |
| `down_l2_mean` | 0.5312 ± 0.0104 | 0.5326 ± 0.0224 |
| `down_l1_mean` | 0.5295 ± 0.0104 | 0.5285 ± 0.0224 |
| `down_linf_mean` | 0.5278 ± 0.0104 | 0.5218 ± 0.0224 |
| `down_jsd_mean` | 0.5243 ± 0.0104 | 0.5102 ± 0.0224 |
| `down_l2_max` | 0.5130 ± 0.0104 | 0.4956 ± 0.0222 |
| `down_l1_max` | 0.5095 ± 0.0104 | 0.4946 ± 0.0223 |
| `down_linf_max` | 0.5048 ± 0.0104 | 0.4780 ± 0.0218 |
| `down_jsd_max` | 0.4935 ± 0.0104 | 0.4719 ± 0.0220 |

## Paired McNemar — best nuc-dep score vs baselines (Global level)

Best nuc-dep is picked per dataset by Global PA. Test is two-sided, BH-adjusted within dataset (2 tests each).

| dataset | candidate (best nuc-dep) | baseline | PA(cand) | PA(base) | Δ | n_pairs | n_cand>base | n_base>cand | p | q |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mendelian_traits | `down_jsd_mean` | `minus_llr` | 0.7320 | 0.7513 | -0.0193 | 4910 | 307 | 402 | 0.0004 | 0.0008 ★ |
| mendelian_traits | `down_jsd_mean` | `embed_l2_flat_last` | 0.7320 | 0.7424 | -0.0104 | 4910 | 152 | 203 | 0.0079 | 0.0079 ★ |
| complex_traits | `down_linf_mean` | `abs_llr` | 0.5762 | 0.5372 | +0.0390 | 564 | 120 | 98 | 0.1548 | 0.3096 |
| complex_traits | `down_linf_mean` | `embed_l2_flat_last` | 0.5762 | 0.5674 | +0.0089 | 564 | 52 | 47 | 0.6879 | 0.6879 |
| eqtl | `down_l2_mean` | `abs_llr` | 0.5312 | 0.5106 | +0.0206 | 2306 | 547 | 500 | 0.1551 | 0.3102 |
| eqtl | `down_l2_mean` | `embed_l2_flat_last` | 0.5312 | 0.5225 | +0.0087 | 2306 | 275 | 255 | 0.4092 | 0.4092 |

★ = q < 0.05.

## Spearman correlation — is nuc-dep more like LLR or L2?

Both within-subset average and overall Global across all variants.

| dataset | comparison | Spearman (global) | Spearman (within-subset mean) |
|---|---|---:|---:|
| mendelian_traits | `down_jsd_mean` × `minus_llr` | 0.896 | 0.608 |
| mendelian_traits | `down_jsd_mean` × `embed_l2_flat_last` | 0.975 | 0.898 |
| mendelian_traits | `minus_llr` × `embed_l2_flat_last` | 0.893 | 0.599 |
| complex_traits | `down_linf_mean` × `abs_llr` | 0.391 | 0.505 |
| complex_traits | `down_linf_mean` × `embed_l2_flat_last` | 0.853 | 0.850 |
| complex_traits | `abs_llr` × `embed_l2_flat_last` | 0.281 | 0.456 |
| eqtl | `down_l2_mean` × `abs_llr` | 0.163 | 0.380 |
| eqtl | `down_l2_mean` × `embed_l2_flat_last` | 0.736 | 0.802 |
| eqtl | `abs_llr` × `embed_l2_flat_last` | -0.022 | 0.248 |

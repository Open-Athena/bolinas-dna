🤖 **Iter-6 — 3-way: LLR-default vs `down_jsd_mean` vs `mean_rank(LLR, L2)` ensemble.**

All FWD+RC AVG, exp166-p1B. `mean_rank` = within-subset average rank of LLR-default (`minus_llr` for mendelian, `abs_llr` for complex/eqtl) and `embed_l2_flat_last` — the iter-5 standard ensemble. Paired McNemar (two-sided) at Global level, BH within dataset (3 tests each).

## Global PA

| dataset | LLR | down_jsd_mean | mean_rank(LLR, L2) |
|---|---:|---:|---:|
| mendelian_traits | 0.7513 ± 0.0062 | 0.7320 ± 0.0063 | 0.7588 ± 0.0061 |
| complex_traits | 0.5372 ± 0.0210 | 0.5709 ± 0.0208 | 0.5674 ± 0.0209 |
| eqtl | 0.5106 ± 0.0104 | 0.5243 ± 0.0104 | 0.5147 ± 0.0104 |

## Pairwise paired tests (Global level)

Δ = PA(A) − PA(B); positive favors A.

### mendelian_traits

| A vs B | PA(A) | PA(B) | Δ | n_pairs | n_A>B | n_B>A | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_jsd_mean` vs `LLR` | 0.7320 | 0.7513 | -0.0193 | 4910 | 307 | 402 | 0.0004 | 0.0006 ★ |
| `mean_rank` vs `LLR` | 0.7588 | 0.7513 | +0.0074 | 4910 | 205 | 167 | 0.0549 | 0.0549 |
| `mean_rank` vs `down_jsd_mean` | 0.7588 | 0.7320 | +0.0268 | 4910 | 277 | 146 | 1.9e-10 | 5.6e-10 ★ |

### complex_traits

| A vs B | PA(A) | PA(B) | Δ | n_pairs | n_A>B | n_B>A | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_jsd_mean` vs `LLR` | 0.5709 | 0.5372 | +0.0337 | 564 | 110 | 91 | 0.2041 | 0.3062 |
| `mean_rank` vs `LLR` | 0.5674 | 0.5372 | +0.0301 | 564 | 69 | 52 | 0.1455 | 0.3062 |
| `mean_rank` vs `down_jsd_mean` | 0.5674 | 0.5709 | -0.0035 | 564 | 67 | 69 | 0.9317 | 0.9317 |

### eqtl

| A vs B | PA(A) | PA(B) | Δ | n_pairs | n_A>B | n_B>A | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_jsd_mean` vs `LLR` | 0.5243 | 0.5106 | +0.0137 | 2306 | 538 | 507 | 0.3534 | 0.6216 |
| `mean_rank` vs `LLR` | 0.5147 | 0.5106 | +0.0041 | 2306 | 292 | 282 | 0.7072 | 0.7072 |
| `mean_rank` vs `down_jsd_mean` | 0.5147 | 0.5243 | -0.0095 | 2306 | 320 | 342 | 0.4144 | 0.6216 |

★ = q < 0.05.

## Per-subset PA

### mendelian_traits

| subset | n_pairs | LLR | down_jsd_mean | mean_rank |
|---|---:|---:|---:|---:|
| missense_variant | 4495 | 0.7548 | 0.7328 | 0.7594 |
| 5_prime_UTR_variant | 87 | 0.7011 | 0.6552 | 0.6954 |
| splicing | 78 | 0.8077 | 0.8590 | 0.8718 |
| tss_proximal | 61 | 0.6557 | 0.5574 | 0.6230 |
| 3_prime_UTR_variant | 58 | 0.6552 | 0.6207 | 0.7500 |
| distal | 56 | 0.6964 | 0.8393 | 0.7500 |
| non_coding_transcript_exon_variant | 42 | 0.6429 | 0.8333 | 0.7500 |
| synonymous_variant | 33 | 0.8485 | 0.7273 | 0.8636 |

### complex_traits

| subset | n_pairs | LLR | down_jsd_mean | mean_rank |
|---|---:|---:|---:|---:|
| distal | 428 | 0.5327 | 0.5491 | 0.5584 |
| missense_variant | 57 | 0.6842 | 0.7368 | 0.6930 |
| tss_proximal | 29 | 0.4483 | 0.5517 | 0.4828 |
| 3_prime_UTR_variant | 21 | 0.5238 | 0.7143 | 0.6190 |
| non_coding_transcript_exon_variant | 12 | 0.4167 | 0.5833 | 0.6250 |
| 5_prime_UTR_variant | 10 | 0.5000 | 0.6000 | 0.5000 |
| synonymous_variant | 6 | 0.3333 | 0.1667 | 0.3333 |
| splicing | 1 | 0.0000 | 0.0000 | 0.0000 |

### eqtl

| subset | n_pairs | LLR | down_jsd_mean | mean_rank |
|---|---:|---:|---:|---:|
| distal | 1678 | 0.5051 | 0.5250 | 0.4979 |
| tss_proximal | 241 | 0.5519 | 0.5270 | 0.5768 |
| non_coding_transcript_exon_variant | 157 | 0.4841 | 0.5350 | 0.5860 |
| 3_prime_UTR_variant | 115 | 0.5652 | 0.4783 | 0.5130 |
| 5_prime_UTR_variant | 51 | 0.4706 | 0.5294 | 0.5294 |
| missense_variant | 30 | 0.4667 | 0.4667 | 0.4667 |
| synonymous_variant | 29 | 0.5172 | 0.6897 | 0.6379 |
| splicing | 5 | 0.6000 | 0.2000 | 0.4000 |

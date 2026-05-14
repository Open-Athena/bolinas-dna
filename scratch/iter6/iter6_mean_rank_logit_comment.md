🤖 **Iter-6 — logit-space ensemble: `mean_rank(LLR, down_jsd_mean)` vs the 3 existing candidates.**

All FWD+RC AVG on exp166-p1B. New candidate stays entirely in logit space (no embeddings). Paired McNemar (two-sided) at Global level, BH within dataset (3 tests each — new ensemble vs each existing candidate).

## Global PA — all 4 candidates

| dataset | LLR | down_jsd_mean | mean_rank(LLR, L2) | **mean_rank(LLR, jsd_mean)** |
|---|---:|---:|---:|---:|
| mendelian_traits | 0.7513 | 0.7320 | 0.7588 | **0.7515** |
| complex_traits | 0.5372 | 0.5709 | 0.5674 | **0.5771** |
| eqtl | 0.5106 | 0.5243 | 0.5147 | **0.5278** |

## Paired tests — `mean_rank(LLR, jsd_mean)` vs each existing candidate

Δ = PA(new) − PA(baseline); positive favors the new logit-space ensemble.

### mendelian_traits

| vs baseline | PA(new) | PA(base) | Δ | n_pairs | n_new>base | n_base>new | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `LLR` | 0.7515 | 0.7513 | +0.0002 | 4910 | 184 | 183 | 1.0000 | 1.0000 |
| `down_jsd_mean` | 0.7515 | 0.7320 | +0.0196 | 4910 | 218 | 122 | 2.2e-07 | 6.5e-07 ★ |
| `mean_rank_LLR_L2` | 0.7515 | 0.7588 | -0.0072 | 4910 | 77 | 111 | 0.0159 | 0.0238 ★ |

### complex_traits

| vs baseline | PA(new) | PA(base) | Δ | n_pairs | n_new>base | n_base>new | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `LLR` | 0.5771 | 0.5372 | +0.0399 | 564 | 60 | 38 | 0.0334 | 0.1001 |
| `down_jsd_mean` | 0.5771 | 0.5709 | +0.0062 | 564 | 52 | 48 | 0.7644 | 0.7644 |
| `mean_rank_LLR_L2` | 0.5771 | 0.5674 | +0.0098 | 564 | 40 | 35 | 0.6445 | 0.7644 |

### eqtl

| vs baseline | PA(new) | PA(base) | Δ | n_pairs | n_new>base | n_base>new | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `LLR` | 0.5278 | 0.5106 | +0.0171 | 2306 | 271 | 232 | 0.0901 | 0.2233 |
| `down_jsd_mean` | 0.5278 | 0.5243 | +0.0035 | 2306 | 271 | 263 | 0.7620 | 0.7620 |
| `mean_rank_LLR_L2` | 0.5278 | 0.5147 | +0.0130 | 2306 | 189 | 161 | 0.1489 | 0.2233 |

★ = q < 0.05.

## Per-subset PA

### mendelian_traits

| subset | n_pairs | LLR | jsd_mean | mean_rank(LLR,L2) | **mean_rank(LLR,jsd_mean)** |
|---|---:|---:|---:|---:|---:|
| missense_variant | 4495 | 0.7548 | 0.7328 | 0.7594 | **0.7526** |
| 5_prime_UTR_variant | 87 | 0.7011 | 0.6552 | 0.6954 | **0.7069** |
| splicing | 78 | 0.8077 | 0.8590 | 0.8718 | **0.8718** |
| tss_proximal | 61 | 0.6557 | 0.5574 | 0.6230 | **0.5574** |
| 3_prime_UTR_variant | 58 | 0.6552 | 0.6207 | 0.7500 | **0.6552** |
| distal | 56 | 0.6964 | 0.8393 | 0.7500 | **0.8036** |
| non_coding_transcript_exon_variant | 42 | 0.6429 | 0.8333 | 0.7500 | **0.7500** |
| synonymous_variant | 33 | 0.8485 | 0.7273 | 0.8636 | **0.8788** |

### complex_traits

| subset | n_pairs | LLR | jsd_mean | mean_rank(LLR,L2) | **mean_rank(LLR,jsd_mean)** |
|---|---:|---:|---:|---:|---:|
| distal | 428 | 0.5327 | 0.5491 | 0.5584 | **0.5689** |
| missense_variant | 57 | 0.6842 | 0.7368 | 0.6930 | **0.7018** |
| tss_proximal | 29 | 0.4483 | 0.5517 | 0.4828 | **0.4655** |
| 3_prime_UTR_variant | 21 | 0.5238 | 0.7143 | 0.6190 | **0.6429** |
| non_coding_transcript_exon_variant | 12 | 0.4167 | 0.5833 | 0.6250 | **0.5833** |
| 5_prime_UTR_variant | 10 | 0.5000 | 0.6000 | 0.5000 | **0.6000** |
| synonymous_variant | 6 | 0.3333 | 0.1667 | 0.3333 | **0.3333** |
| splicing | 1 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |

### eqtl

| subset | n_pairs | LLR | jsd_mean | mean_rank(LLR,L2) | **mean_rank(LLR,jsd_mean)** |
|---|---:|---:|---:|---:|---:|
| distal | 1678 | 0.5051 | 0.5250 | 0.4979 | **0.5176** |
| tss_proximal | 241 | 0.5519 | 0.5270 | 0.5768 | **0.5851** |
| non_coding_transcript_exon_variant | 157 | 0.4841 | 0.5350 | 0.5860 | **0.5510** |
| 3_prime_UTR_variant | 115 | 0.5652 | 0.4783 | 0.5130 | **0.5696** |
| 5_prime_UTR_variant | 51 | 0.4706 | 0.5294 | 0.5294 | **0.4314** |
| missense_variant | 30 | 0.4667 | 0.4667 | 0.4667 | **0.4667** |
| synonymous_variant | 29 | 0.5172 | 0.6897 | 0.6379 | **0.5862** |
| splicing | 5 | 0.6000 | 0.2000 | 0.4000 | **0.5000** |

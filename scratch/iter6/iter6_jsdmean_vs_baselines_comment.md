🤖 **Iter-6 — fix `down_jsd_mean` (systematic best from previous comment) and compare to LLR / L2 baselines.**

Paired McNemar (two-sided) at Global level, BH within dataset (2 tests each). Δ = PA(`down_jsd_mean`) − PA(baseline).

## Global paired tests

| dataset | baseline | PA(`down_jsd_mean`) | PA(baseline) | Δ | n_pairs | n_cand>base | n_base>cand | p | q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mendelian_traits | `minus_llr` | 0.7320 | 0.7513 | -0.0193 | 4910 | 307 | 402 | 0.0004 | 0.0008 ★ |
| mendelian_traits | `embed_l2_flat_last` | 0.7320 | 0.7424 | -0.0104 | 4910 | 152 | 203 | 0.0079 | 0.0079 ★ |
| complex_traits | `abs_llr` | 0.5709 | 0.5372 | +0.0337 | 564 | 110 | 91 | 0.2041 | 0.4082 |
| complex_traits | `embed_l2_flat_last` | 0.5709 | 0.5674 | +0.0035 | 564 | 58 | 56 | 0.9254 | 0.9254 |
| eqtl | `abs_llr` | 0.5243 | 0.5106 | +0.0137 | 2306 | 538 | 507 | 0.3534 | 0.7068 |
| eqtl | `embed_l2_flat_last` | 0.5243 | 0.5225 | +0.0017 | 2306 | 268 | 264 | 0.8965 | 0.8965 |

★ = q < 0.05.

## Spearman correlation (within-subset mean)

| dataset | `down_jsd_mean` × LLR | `down_jsd_mean` × L2 |
|---|---:|---:|
| mendelian_traits | 0.608 | 0.898 |
| complex_traits | 0.534 | 0.844 |
| eqtl | 0.376 | 0.778 |

## Per-subset PA

### mendelian_traits

| subset | n_pairs | `down_jsd_mean` | `minus_llr` | `embed_l2_flat_last` |
|---|---:|---:|---:|---:|
| missense_variant | 4495 | 0.7328 | 0.7548 | 0.7419 |
| 5_prime_UTR_variant | 87 | 0.6552 | 0.7011 | 0.6897 |
| splicing | 78 | 0.8590 | 0.8077 | 0.8718 |
| tss_proximal | 61 | 0.5574 | 0.6557 | 0.6066 |
| 3_prime_UTR_variant | 58 | 0.6207 | 0.6552 | 0.7241 |
| distal | 56 | 0.8393 | 0.6964 | 0.8214 |
| non_coding_transcript_exon_variant | 42 | 0.8333 | 0.6429 | 0.7857 |
| synonymous_variant | 33 | 0.7273 | 0.8485 | 0.7273 |

### complex_traits

| subset | n_pairs | `down_jsd_mean` | `abs_llr` | `embed_l2_flat_last` |
|---|---:|---:|---:|---:|
| distal | 428 | 0.5491 | 0.5327 | 0.5561 |
| missense_variant | 57 | 0.7368 | 0.6842 | 0.7018 |
| tss_proximal | 29 | 0.5517 | 0.4483 | 0.5862 |
| 3_prime_UTR_variant | 21 | 0.7143 | 0.5238 | 0.5238 |
| non_coding_transcript_exon_variant | 12 | 0.5833 | 0.4167 | 0.5000 |
| 5_prime_UTR_variant | 10 | 0.6000 | 0.5000 | 0.6000 |
| synonymous_variant | 6 | 0.1667 | 0.3333 | 0.3333 |
| splicing | 1 | 0.0000 | 0.0000 | 0.0000 |

### eqtl

| subset | n_pairs | `down_jsd_mean` | `abs_llr` | `embed_l2_flat_last` |
|---|---:|---:|---:|---:|
| distal | 1678 | 0.5250 | 0.5051 | 0.5173 |
| tss_proximal | 241 | 0.5270 | 0.5519 | 0.5394 |
| non_coding_transcript_exon_variant | 157 | 0.5350 | 0.4841 | 0.5605 |
| 3_prime_UTR_variant | 115 | 0.4783 | 0.5652 | 0.4870 |
| 5_prime_UTR_variant | 51 | 0.5294 | 0.4706 | 0.5490 |
| missense_variant | 30 | 0.4667 | 0.4667 | 0.5000 |
| synonymous_variant | 29 | 0.6897 | 0.5172 | 0.6552 |
| splicing | 5 | 0.2000 | 0.6000 | 0.2000 |

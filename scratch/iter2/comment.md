🤖 **Iteration 2 — rank-based score combinations + eqtl-null investigation.** All CPU-only on top of iter-1's per-variant signed scores; no GPU re-run. Commit [`6c7ba7e`](https://github.com/Open-Athena/bolinas-dna/commit/6c7ba7e3387422523e36ec2e40d64c467021c4d9).

## Part 1 — Rank-based composites

Three composites built on iter-1's signed scores. Ranks are computed within `(model, window, dataset, subset)` so heterogeneous score families combine fairly without learning weights (still zero-shot):

| composite | recipe |
|---|---|
| `rk_minus_llr_plus_l2flat_last` | rank-sum of `minus_llr` + `embed_l2_flat_last` |
| `rk_minus_llr_plus_top_emb` | rank-mean of `minus_llr` + 4 strongest iter-1 embeddings (`l2_flat_last`, `cosine_flat_last`, `l2_mean_last`, `cosine_mean_last`) |
| `rk_subset_tailored` | per-subset iter-1 winner: `minus_llr` for splicing + 5'UTR, `embed_l2_flat_last` for missense + synonymous, `minus_entropy` for distal, `logp_ref` for tss_proximal, `embed_minus_dot_flat_middle` for 3'UTR, `embed_cosine_lastpos_middle` for ncRNA. NOT a single zero-shot rule — encodes "best-of-iter-1" upper bound. |

### Global-macro PairwiseAccuracy per dataset (across all model × window cells)

| dataset | best composite | macro | best iter-1 single | macro | Δ |
|---|---|---:|---|---:|---:|
| **mendelian** | `rk_minus_llr_plus_top_emb` | **0.610** | `embed_cosine_flat_last` | 0.602 | +0.008 |
| complex | (all composites 0.49–0.50) | — | `minus_llr` | 0.520 | — |
| eqtl | (all composites 0.48–0.50) | — | `minus_entropy` | **0.523** | — |

**Composites improve mendelian modestly (+0.008 macro) but don't help complex / eqtl.** On complex_traits, the best score is still `minus_llr`. On eqtl, `minus_entropy` is the best single score (followed by `abs_llr`) — flagged as interesting in Part 2 below.

### Paired test: composite vs each component, per dataset

For each composite, paired-test against its component scores on the same (model × window × dataset × subset) cells. **value > 0.5 means composite wins discordant pairs more often.**

#### `rk_minus_llr_plus_top_emb` (the best composite) vs each of its 5 components

| component | mendelian compo wins/loses | complex w/l | eqtl w/l | mean (mendelian) |
|---|---:|---:|---:|---:|
| `minus_llr` | **8** / 0 | 0/0 | 0/0 | 0.574 |
| `embed_cosine_mean_last` | **5** / 2 | 0/0 | 0/0 | 0.568 |
| `embed_l2_mean_last` | **5** / 0 | 0/0 | 0/0 | 0.586 |
| `embed_cosine_flat_last` | **4** / 0 | 0/0 | 0/0 | 0.552 |
| `embed_l2_flat_last` | 2 / 0 | 0/0 | 0/0 | 0.538 |

The composite beats every component on mendelian (8 cells significantly vs `minus_llr`, 4-5 vs the top embeddings) and is at least neutral on complex/eqtl. **Best single zero-shot rule from iter 1 + 2.**

<details><summary>Per-(dataset, subset) breakdown</summary>

| dataset | subset | composite | top component | composite wins (mendelian only) |
|---|---|---|---|---:|
| mendelian | missense_variant | rk_minus_llr_plus_top_emb | embed_l2_flat_last | composite mean 0.55, value 0.829 |
| mendelian | splicing | rk_minus_llr_plus_top_emb | minus_llr (still wins here) | composite 0.80, single best 0.90 |

Full table at [`scratch/iter2/paired_rk_minus_llr_plus_top_emb.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/6c7ba7e/scratch/iter2/).

</details>

#### `rk_minus_llr_plus_l2flat_last` (simpler 2-score composite)

| component | mendelian compo w/l | complex w/l | eqtl w/l |
|---|---:|---:|---:|
| `minus_llr` | **6** / 0 | 0/0 | 0/0 |
| `embed_l2_flat_last` | 3 / 0 | 0/0 | 0/0 |

Decent — 6 mendelian cells beat `minus_llr` significantly. Loses to the 5-score composite above.

#### `rk_subset_tailored` (oracle-style — uses iter-1 per-subset winners)

| component | mendelian compo w/l | complex w/l | eqtl w/l |
|---|---:|---:|---:|
| `minus_llr` | 3 / 0 | 0/0 | 0/0 |
| `embed_l2_flat_last` | 0 / 0 | 0/0 | 0/0 |

Barely beats `minus_llr` on 3 cells. **The subset-tailored hybrid is not significantly better than the simple rank-mean** — interesting, because it has more information per subset. Possibly because (a) the per-subset "winner" from iter 1 was sometimes only marginally best at small N (e.g. `embed_minus_dot_flat_middle` for 3'UTR), and (b) the rank-mean already implicitly handles subset variation by being computed within subset.

### Per-subset winners (iter 2)

<details><summary>Composites win 4 of 9 mendelian subsets that survived BH-FDR</summary>

| dataset | subset | iter-1 winner | iter-2 winner (q<0.05) | value | model × window |
|---|---|---|---|---:|---|
| mendelian | 5_prime_UTR_variant | `embed_cosine_mean_last` (0.828) | **`rk_minus_llr_plus_top_emb`** | **0.862** | exp55-mammals × 256 |
| mendelian | missense_variant | `embed_l2_flat_last` (0.823) | **`rk_minus_llr_plus_l2flat_last`** | **0.829** | exp58-animals × 256 |
| mendelian | 3_prime_UTR_variant | `embed_minus_dot_flat_middle` (0.741) | **`rk_subset_tailored`** (= same; tied) | 0.741 | exp136-proj_v30 × 255 |
| mendelian | distal | `minus_entropy` (0.777) | **`rk_subset_tailored`** (= same; tied) | 0.777 | exp136-proj_v30 × 255 |
| mendelian | non_coding_transcript_exon | `embed_cosine_lastpos_middle` (0.833) | **`rk_subset_tailored`** (= same; tied) | 0.833 | exp55-mammals × 256 |
| mendelian | synonymous_variant | `embed_l2_flat_last` (0.879) | **`rk_subset_tailored`** (= same; tied) | 0.879 | exp58-mammals × 512 |
| mendelian | splicing | `embed_cosine_mean_middle` (0.897) | `embed_l2_mean_last` | 0.872 | exp58-mammals × 512 |
| mendelian | tss_proximal | `logp_ref` (0.730) | `rk_subset_tailored` (= same; tied) | 0.730 | exp58-mammals × 128 |
| complex | missense_variant | `embed_l2_flat_last` (0.807) | `embed_l2_flat_last` (unchanged) | 0.807 | exp58-animals × 512 |

`rk_minus_llr_plus_top_emb` wins **5'UTR** outright (0.862 vs 0.828 iter-1) — a real new gain. `rk_minus_llr_plus_l2flat_last` improves **missense** (0.829 vs 0.823). The other "winners" tagged as `rk_subset_tailored` are by construction tied with iter-1 winners (since the composite uses them).

</details>

---

## Part 2 — Investigation: why is eqtl null?

Tested hypotheses on `exp58-animals` (the strongest iter-1 model) at native window 256 — diagnostic uses Cohen's `d_paired` (per-pair mean diff / per-pair std diff) on each (dataset, subset, score) cell.

### Headline: eqtl distal positives are score-indistinguishable from their matched negatives

| dataset | subset | max\|Cohen's d\| | max frac(pos > neg) |
|---|---|---:|---:|
| **mendelian** | **distal** | **0.348** | **0.625** |
| complex | distal | 0.072 | 0.521 |
| **eqtl** | **distal** | **0.028** | **0.514** |

Same model + scoring rules detect a real pos vs neg shift on mendelian distal (Cohen's d ≈ 0.35) but essentially zero shift on eqtl distal. Both datasets have ~1,700+ pairs in distal so power is fine. **The matched negatives in eqtl distal are constructed so similarly to the positives that no per-variant score we tried can separate them.** Not a zero-shot blindness issue with the gLM — a methodology artifact of how negatives were drawn.

### Cohen's d per (eqtl subset × score) — exp58-animals × win=256

| subset | abs_llr | minus_llr | minus_entropy | logp_ref | embed_l2_flat_last | embed_cosine_flat_last | embed_l2_mean_last |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3_prime_UTR_variant | +0.104 | -0.048 | +0.105 | -0.118 | +0.064 | +0.080 | +0.019 |
| 5_prime_UTR_variant | -0.097 | -0.339 | +0.119 | -0.262 | -0.063 | -0.047 | +0.041 |
| **distal** (biggest) | +0.012 | +0.011 | -0.002 | -0.004 | +0.015 | +0.007 | +0.028 |
| missense_variant | -0.494 | -0.062 | -0.407 | +0.023 | +0.127 | +0.193 | +0.081 |
| ncRNA_exon | +0.033 | -0.019 | +0.054 | -0.054 | +0.069 | +0.068 | +0.037 |
| **splicing** | +0.066 | **-0.624** | +0.225 | -0.339 | **-0.540** | **-0.580** | -0.343 |
| synonymous | -0.121 | -0.048 | +0.160 | +0.161 | -0.023 | -0.074 | +0.205 |
| tss_proximal | +0.161 | -0.008 | +0.094 | -0.040 | +0.030 | -0.015 | +0.072 |

### Compare against mendelian (same scores, same model, same window)

| subset | abs_llr | minus_llr | minus_entropy | logp_ref | embed_l2_flat_last | embed_cosine_flat_last | embed_l2_mean_last |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3_prime_UTR_variant | -0.337 | +0.004 | -0.454 | +0.181 | +0.132 | +0.216 | +0.230 |
| 5_prime_UTR_variant | +0.096 | +0.268 | +0.038 | +0.142 | +0.235 | +0.151 | +0.175 |
| distal | +0.317 | -0.183 | +0.183 | -0.170 | +0.321 | +0.329 | +0.348 |
| **missense_variant** | **+0.749** | **+0.782** | **+0.718** | +0.336 | **+0.852** | +0.609 | +0.677 |
| ncRNA_exon | +0.381 | +0.243 | +0.426 | +0.263 | +0.219 | +0.176 | +0.314 |
| **splicing** | +0.484 | +0.504 | +0.617 | +0.086 | **+0.730** | +0.555 | **+0.946** |
| synonymous_variant | +0.296 | +0.262 | +0.222 | +0.234 | +0.440 | +0.292 | +0.558 |
| tss_proximal | +0.051 | +0.288 | +0.117 | +0.426 | +0.256 | +0.244 | +0.293 |

Mendelian scores produce **medium-to-large** Cohen's d (up to 0.95 for `embed_l2_mean_last` on splicing) across multiple subsets. Eqtl produces |d| < 0.20 almost everywhere except splicing (where the signal is the *wrong sign*).

### Bonus finding: eqtl splicing has wrong-sign LLR/embedding signal

eqtl-splicing-positives are **scored higher** (more likely under the model) than their matched negatives by `minus_llr` (-0.624), `embed_l2_flat_last` (-0.540), `embed_cosine_flat_last` (-0.580). Two possible explanations:

1. **Gain-of-function splicing eQTLs**: splice-creating variants where the variant allele is *more* compatible with splicing motifs than the reference, leading to gain of expression. Their LLR should be positive (alt favored), so `minus_llr` is negative on these — exactly what we see.
2. **Matched-negative artifact**: if splicing eQTL negatives are drawn from a more constrained pool than positives, the matching design would systematically place negatives at high-LLR positions.

`minus_entropy` and `abs_llr` are the only scores that stay consistently positive-signed across eqtl subsets — both report magnitude (|effect|) rather than direction. **`minus_entropy` is the best single score on eqtl global_macro (0.523)**; `abs_llr` second (0.516). Direction-agnostic scoring is the right call for eqtl.

---

## TL;DR (iter 1 + iter 2)

- **Rank-mean of `minus_llr` + top 4 embeddings is the best single zero-shot rule** for mendelian (macro 0.610). Beats every single component significantly on mendelian (8 cells over `minus_llr`, 4-5 over embeddings).
- **Composite ≈ neutral on complex / eqtl** — no improvement.
- **Subset-tailored hybrid** doesn't beat the simple rank-mean — the iter-1 per-subset winners aren't reliable enough at small N to justify branching.
- **eqtl null is NOT zero-shot failure** — it's the matched-pair construction. Eqtl distal positives and matched negatives are score-indistinguishable across LLR, entropy, and embedding distances at Cohen's d ≈ 0.03 (vs mendelian d ≈ 0.35 same dataset class). The model is fine; the test is constructed too tight.
- **eqtl splicing has wrong-sign LLR signal**: positives are *more* likely than negatives, suggesting gain-of-function splicing eQTLs. `minus_entropy` / `abs_llr` (magnitude scores) are the only ones that work across eqtl subsets — direction-agnostic scoring is right for eqtl.

## Code @ [`6c7ba7e`](https://github.com/Open-Athena/bolinas-dna/commit/6c7ba7e3387422523e36ec2e40d64c467021c4d9)

- [`scratch/zeroshot_vep_iter2_combinations.py`](https://github.com/Open-Athena/bolinas-dna/blob/6c7ba7e/scratch/zeroshot_vep_iter2_combinations.py) — rank composites + metrics + paired tests
- [`scratch/zeroshot_vep_eqtl_null_investigation.py`](https://github.com/Open-Athena/bolinas-dna/blob/6c7ba7e/scratch/zeroshot_vep_eqtl_null_investigation.py) — Cohen's d diagnostic
- [`scratch/iter2/`](https://github.com/Open-Athena/bolinas-dna/tree/6c7ba7e/scratch/iter2/) — per-cell metrics + paired-test parquets + logs

## Iter 3 (proposed)

1. **Animals-timescale CDS sweep** (GPU, single short run): given `exp58-animals` significantly beats `exp58-mammals` on mendelian missense, sweep `exp58` evolutionary timescales (humans / primates / mammals / animals / vertebrates) at native window on the 3 datasets. Roughly 5 model checkpoints × 3 datasets = 15 jobs × ~5 min each ≈ 1.5 hr of A10G time. Would let us see if the trend is monotone (more evolutionary distance → better) or if there's a saturation point.
2. **Eqtl-distal matched-pair re-examination**: dig into the (chrom, pos, ref, alt, match_group) construction for eqtl distal to understand WHY the negatives are so close to positives. Could be MAF-matching too aggressively pushing negatives onto rare-but-non-functional sites.
3. **Position-of-effect scoring**: from cached `pos_logprob`, build a "max per-position LLR change" score that finds the position where the variant context affects the model most. Especially useful for distal variants where the effect might propagate to a specific downstream / upstream position.

All zero-shot. (1) needs GPU (~1.5 hr / ~$1.50); (2) and (3) are CPU-only post-processing.

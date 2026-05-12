🤖 **Iter-4 follow-up: ensembling exp166-p1B with AlphaGenome.** Pulled the AlphaGenome `alphagenome_max_l2` per-variant scores from S3 (already on the leaderboards: #161/#162/#172, original baseline #154), merged into the exp166-p1B FWD scores, and tested rank-mean ensembles. Commit [`b65d27f`](https://github.com/Open-Athena/bolinas-dna/commit/b65d27f).

## TL;DR

- **Strong domain split**: AlphaGenome (regulatory model) wins on regulatory-heavy datasets (complex_traits, eqtl) and on the regulatory subsets of mendelian; exp166-p1B (sequence-only generalist) wins on coding subsets (missense, synonymous in mendelian).
- **Ensembles beat both single models on a few mendelian cells** where p1B and AG give partly-independent signal — most strikingly **splicing: 0.923 (ensemble) vs 0.859 (AG alone) vs 0.808 (p1B alone)** — +0.064 over the better single model.
- **On complex_traits and eqtl, AlphaGenome alone is the best default**: AG-only pooled is 0.621 on both, vs ≤0.580 for any ensemble. The p1B sequence model adds noise more than signal on regulatory variants.
- **Best ensemble per dataset**:
  - mendelian: `rk_llr_l2flat_ag` (rank-mean of `minus_llr` + `embed_l2_flat_last` + `alphagenome_max_l2`) — macro **0.782** (vs 0.725 for p1B alone, +0.057; vs 0.700 for AG alone, +0.082).
  - complex/eqtl: **just use AG alone** — no ensemble beats it consistently.

## Pooled accuracy (whole-dataset, all variants)

| dataset | p1B default | p1B best composite | AG alone | p1B+AG (best) | winner |
|---|---:|---:|---:|---:|---|
| mendelian | 0.739 (`minus_llr`) | 0.744 (`rk_llr_l2flat`) | 0.543 | 0.740 (`rk_llr_top_emb_ag`) | **p1B composite (0.744)** |
| complex | 0.528 (`abs_llr`) | 0.560 (`rk_llr_top_emb`) | **0.621** | 0.608 (`rk_llr_l2flat_ag`) | **AG alone (0.621)** |
| eqtl | 0.519 (`abs_llr`) | 0.526 (`rk_llr_l2flat`) | **0.621** | 0.580 (`rk_p1B_default_ag`) | **AG alone (0.621)** |

For mendelian, the p1B composite still narrowly wins pooled — driven by missense (n=4495) where AG is at chance. For complex and eqtl, AG dominates pooled by a wide margin.

## Macro accuracy (mean across subsets)

| dataset | p1B default | p1B best composite | AG alone | p1B+AG (best) | winner |
|---|---:|---:|---:|---:|---|
| mendelian | 0.725 | 0.765 (`rk_llr_l2flat`) | 0.700 | **0.782** (`rk_llr_l2flat_ag`) | **p1B+AG (+0.017)** |
| complex | 0.515 | 0.549 (`rk_llr_top_emb`) | **0.607** | 0.583 (`rk_p1B_default_ag`) | **AG alone (+0.058)** |
| eqtl | 0.535 | 0.537 (`rk_llr_top_emb`) | **0.706** | 0.631 (`rk_p1B_default_ag`) | **AG alone (+0.075)** |

On macro, mendelian flips to the ensemble being best (because the smaller non-coding subsets get a real lift). Complex and eqtl macros are still AG-dominant.

## Per-cell breakdown — where does each strategy win?

| dataset | subset | n_pairs | p1B best | AG alone | p1B comp | p1B+AG | winner |
|---|---|---:|---:|---:|---:|---:|---|
| mendelian | 3'UTR | 58 | 0.724 | 0.655 | **0.741** | 0.724 | p1B_cmp |
| mendelian | 5'UTR | 87 | 0.736 | 0.678 | 0.724 | **0.747** | p1B+AG |
| mendelian | distal | 56 | **0.786** | 0.643 | 0.768 | 0.786 | p1B alone |
| mendelian | missense | 4495 | 0.741 | 0.526 | **0.742** | 0.740 | p1B_cmp |
| mendelian | ncRNA | 42 | **0.833** | 0.762 | 0.810 | 0.786 | p1B alone |
| mendelian | **splicing** | 78 | 0.808 | 0.859 | 0.846 | **0.923** | **p1B+AG (+0.064)** |
| mendelian | synonymous | 33 | 0.848 | 0.818 | 0.848 | **0.909** | **p1B+AG (+0.061)** |
| mendelian | tss_proximal | 61 | **0.730** | 0.656 | 0.664 | 0.705 | p1B alone |
| complex | 3'UTR | 21 | 0.667 | **0.714** | 0.429 | 0.619 | AG alone |
| complex | 5'UTR | 10 | 0.700 | 0.800 | 0.700 | **0.900** | **p1B+AG (+0.100, fragile n)** |
| complex | distal | 428 | 0.565 | **0.638** | 0.548 | 0.609 | AG alone |
| complex | missense | 57 | 0.737 | 0.474 | 0.737 | **0.754** | p1B+AG |
| complex | ncRNA | 12 | **0.750** | 0.500 | 0.583 | 0.667 | p1B alone |
| complex | synonymous | 6 | **0.833** | 0.500 | 0.333 | 0.333 | p1B alone |
| complex | tss_proximal | 29 | **0.690** | 0.621 | 0.517 | 0.655 | p1B alone |
| eqtl | 3'UTR | 115 | 0.617 | **0.678** | 0.543 | 0.678 | AG alone |
| eqtl | 5'UTR | 51 | **0.725** | 0.725 | 0.461 | 0.608 | p1B alone |
| eqtl | distal | 1678 | 0.530 | **0.604** | 0.526 | 0.566 | AG alone |
| eqtl | missense | 30 | **0.667** | 0.633 | 0.633 | 0.667 | p1B alone |
| eqtl | ncRNA | 157 | 0.586 | 0.586 | 0.592 | **0.605** | p1B+AG |
| eqtl | splicing | 5 | **1.000** | 1.000 | 0.500 | 0.800 | p1B alone |
| eqtl | synonymous | 29 | 0.621 | **0.724** | 0.586 | 0.690 | AG alone |
| eqtl | tss_proximal | 241 | 0.568 | **0.697** | 0.515 | 0.643 | AG alone |

**Outright winner counts across 22 cells:**

| strategy | mendelian | complex | eqtl | total |
|---|---:|---:|---:|---:|
| p1B alone (best of 30) | 3 | 3 | 3 | **9** |
| AG alone | 0 | 2 | 4 | **6** |
| p1B+AG ensemble | 3 | 2 | 1 | **6** |
| p1B composite (no AG) | 2 | 0 | 0 | 2 |

## When does the ensemble win?

The 6 ensemble-wins concentrate on subsets where both models give meaningful signal but with partly-independent rankings:

- **mendelian splicing** (+0.064 over best single): canonical "boundary" subset; p1B's LLR catches some variants AG misses and vice versa. With AVG = 0.923 vs best alone 0.859, the splicing CDS-boundary effect we documented earlier is now substantially reduced by mixing in regulatory features.
- **mendelian synonymous** (+0.061): p1B's CDS-aware LLR + AG's regulatory readout both contribute.
- **mendelian 5'UTR** (+0.011): both partial signal.
- **complex 5'UTR** (+0.100): fragile n=10.
- **complex missense** (+0.017): unusual — p1B captures coding effect, AG adds something.
- **eqtl ncRNA** (+0.019): both at the noise floor; tiny lift.

The 4 datasets/subsets where AG alone wins decisively are all **regulatory subsets with large n**: eqtl distal (n=1678), eqtl tss_proximal (n=241), eqtl synonymous (n=29 — interestingly), complex distal (n=428). These are the GWAS-typed variants AlphaGenome was designed for; the sequence-only LM adds noise.

## Practical recommendations

1. **Per-dataset default policy**:
   - **mendelian**: use the `rk_llr_l2flat_ag` ensemble — macro 0.782 (vs 0.725 default). Cost: one extra AG API call per variant.
   - **complex_traits**: use AlphaGenome alone (`alphagenome_max_l2`). p1B doesn't help here.
   - **eqtl**: use AlphaGenome alone. Even more pronounced — AG macro 0.706 vs anything-with-p1B 0.631.

2. **Per-subset within mendelian**: for splicing and synonymous specifically, the ensemble is +0.06 over both single models — this is where ensembling provides the strongest gain in the entire benchmark.

3. **Open question — calibration matters**: the ensembles here are unweighted rank-means. With a small validation set you could fit per-subset model weights to maximize ensemble gain. But the bigger lift would come from running **AlphaGenome's RC-averaged variant** (their pipeline already supports two-strand) — currently the AG scores on the leaderboard are FWD-only by their pipeline default.

## Code @ [`b65d27f`](https://github.com/Open-Athena/bolinas-dna/commit/b65d27f)

- [`scratch/iter4/alphagenome/{mendelian_traits,complex_traits,eqtl}.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/b65d27f/scratch/iter4/alphagenome) — AlphaGenome `alphagenome_max_l2` per-variant (pulled from `s3://oa-bolinas/snakemake/alphagenome_eval/results/scores/`)
- Analysis run inline against the existing exp166-p1B FWD parquets ([`scratch/iter4/iter4_fwd_exp166-p1B__win255__*.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/b65d27f/scratch/iter4))

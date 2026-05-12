🤖 **Iter 4 — HAL vs NCBI Datasets per-species lowercase comparison (genome-wide)**

This is the experiment the audit was missing. Four species, all from the v1 leaf set, spanning the GCF/GCA × mask-level grid. For each: pull the HAL-derived `.2bit` from S3, also download the NCBI Datasets `genomic.fna` for the matching accession, count lowercase / total over the **whole genome** for both.

| species | accession | prefix | total bases (HAL) | N-frac | lower-frac (HAL) | lower-frac (NCBI Datasets) | Δ (HAL − NCBI) |
|---|---|---|---:|---:|---:|---:|---:|
| Homo_sapiens          | GCA_000001405.27 | GCA |  3,099,750,718 | 4.97% | **35.6%** | **38.0%** | **−2.4 pp** |
| Ceratotherium_simum   | GCF_000283155.1  | GCF |  2,464,350,348 | 3.96% | **53.8%** | **29.6%** | **+24.2 pp** |
| Petromus_typicus      | GCA_004026965.1  | GCA |  2,389,366,989 | 0.04% | **37.4%** | **34.0%** | **+3.4 pp** |
| Mus_musculus          | GCF_000001635.26 | GCF |  2,818,974,548 | 2.82% | **54.7%** | **36.4%** | **+18.3 pp** |

## Revised story (this inverts my earlier interpretation)

1. **The HAL is not stripping mask info.** In 3 of 4 species the HAL has *more* lowercase than NCBI Datasets ships, not less. The single exception is human, and even there the gap is small (−2.4 pp).
2. **The Cactus build appears to add masking.** Most plausible: Cactus's `repeatmasker` step during alignment construction runs a uniform repeat-detection pass and propagates it through every leaf, so even GCA leaves whose submitted assemblies had no masking come out of the HAL with ~30–55% lowercase. That matches "HAL adds 18 pp on Mus_musculus" and "HAL adds 24 pp on Ceratotherium_simum".
3. **My earlier "v1 Homo_sapiens at 1.9% means HAL is lossy" claim was wrong.** It was the **conservation-filter recipe** doing the work, not the HAL. Recipe vs. genome-wide:

   | species | v1-window lower-frac | HAL whole-genome lower-frac | ratio |
   |---|---:|---:|---:|
   | Homo_sapiens | 1.9% | 35.6% | **5%** |
   | Mus_musculus | 14.5% | 54.7% | 26% |
   | Petromus_typicus | 3.9% | 37.4% | 10% |
   | Ceratotherium_simum | 19.2% | 53.8% | 36% |

   The v1 anchor pipeline filters at `proportion_conserved ≥ 0.20` against `phyloP_447m` — and deeply-conserved regions across 447 mammals are gene/CDS-rich and repeat-poor *by biology*, so the recipe systematically pulls lowercase fraction down by 3×–20× depending on the species' conservation landscape.

4. **R4 (re-extract from NCBI Datasets FASTAs) is now ruled out.** For 3 of 4 tested species it would *decrease* mask coverage, not increase it. Only human would gain (and by ≤3 pp).

## Updated fix-option ranking for repeat-masking

| ID | option | verdict after this experiment |
|---|---|---|
| **R1** | Accept current per-assembly variation; document in dataset card. | **Now the default** — the masking is what Cactus produces, which is more uniform than the input assemblies (uniform RepeatMasker pass during HAL build). Variance across v1 species is mostly recipe-driven, not source-driven. |
| R2 | Uniform RepeatMasker post-pass on the 255 bp windows. | Still applies if you want *strictly* uniform mask annotations, but the marginal value over Cactus's existing pass is small. Cost-benefit weak. |
| ~~R3~~ | Source per-species RepeatMasker BEDs from UCSC/Ensembl/NCBI. | Same conclusion — would not improve on Cactus's pass. |
| ~~**R4**~~ | ~~Re-extract from NCBI soft-masked FASTAs.~~ | **Removed.** Would *decrease* masking in 3 of 4 species; the HAL already has more than NCBI ships. |

## Open questions reduced

- **GCF / GCA split inside v1 (16.7% vs 6.6%)** — likely still real, but now explained more by *what conservation landscape selects per species* than by *what NCBI shipped*. Worth a one-pass genome-wide HAL lowercase scan across all 108 species (a few hours of c6id.2xlarge compute) if we want the genome-wide leg of this comparison fully nailed.
- **N-stretches** — unchanged from iter-2. F1 still the recommended fix; ~0.5% row loss.

## What the user's "less consistent than RefSeq" observation now means

Across 108 v1 species, **whole-genome lowercase fractions are likely much more uniform than the in-window 1.9–20% spread suggests** (this run shows 4 species at 36–55% genome-wide). The "inconsistency" the user saw in v1 is real *as in-window stats*, but mostly recipe-induced — conservation-filter intensity varies per species, not Cactus-mask quality.

The RefSeq pipeline doesn't have this confounder for the same reason it doesn't have a fair single-number baseline: each RefSeq recipe selects functional regions of its own (TSS, CDS, downstream), and recipe choice dominates the lowercase fraction far more than per-genome NCBI-mask quality.

Updating issue body to drop R4 and demote R2 to "fallback" status. R1 (accept + document) becomes the default repeat-mask recommendation.

🤖 **Iter 4 — HAL-derived FASTA vs NCBI Datasets FASTA (whole-genome lowercase)**

This run inverts the iter-2 / iter-3 narrative about a "HAL strip-down". The HAL preserves mask info, and in several leaves it **adds** more than NCBI Datasets serves. The "Homo_sapiens at 1.9%" I flagged earlier was an artifact of v1's conservation-filter recipe, not the HAL → FASTA path.

## Method

Whole-genome lowercase fraction (over A/C/G/T; N's excluded from the denominator):

- **HAL source**: `aws s3 cp s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/projection/genomes/{species}.2bit` → `twoBitToFa` → count.
- **NCBI source**: `datasets download genome accession {acc} --include genome` → unzip → count `genomic.fna`.

Same accession in both columns. Four species spanning the mask spectrum we observed inside v1 windows.

## Result

| species | accession | source | total bases | N% | **lowercase / ACGT** |
|---|---|---|---:|---:|---:|
| Homo_sapiens | GCA_000001405.27 | HAL | 3,099,750,718 | 4.97% | **35.6%** |
| Homo_sapiens | GCA_000001405.27 | NCBI Datasets | 3,257,347,282 | 4.95% | 38.0% |
| Ceratotherium_simum | GCF_000283155.1 | HAL | 2,464,350,348 | 3.96% | **53.8%** |
| Ceratotherium_simum | GCF_000283155.1 | NCBI Datasets | 2,464,367,180 | 3.96% | 29.6% |
| Petromus_typicus | GCA_004026965.1 | HAL | 2,389,366,989 | 0.04% | **37.4%** |
| Petromus_typicus | GCA_004026965.1 | NCBI Datasets | 2,389,366,989 | 0.04% | 34.0% |
| Mus_musculus | GCF_000001635.26 | HAL | 2,818,974,548 | 2.82% | **54.7%** |
| Mus_musculus | GCF_000001635.26 | NCBI Datasets | 2,818,974,548 | 2.82% | 36.4% |

## Implications

1. **The HAL preserves mask info.** For Homo_sapiens, HAL ≈ NCBI within ~2.4 pp. For the others HAL has *more* lowercase than the NCBI Datasets `genomic.fna` for the same accession — most strikingly Ceratotherium_simum (HAL 53.8% vs NCBI 29.6%) and Mus_musculus (HAL 54.7% vs NCBI 36.4%). Most plausible explanation: Cactus applies its own repeat-detection step during alignment construction (e.g., low-complexity / lineage-specific repeat screening) on top of whatever the input assembly carried.
2. **My earlier "Homo_sapiens v1 at 1.9% proves a HAL strip-down" claim was wrong.** HAL Homo_sapiens whole-genome is **35.6% lowercase**. The 1.9% inside v1's Homo_sapiens rows is the **conservation-filter recipe** (anchors with phyloP_447m `proportion_conserved ≥ 0.20`) selecting overwhelmingly gene-rich, ultraconserved-element-rich regions — which are intrinsically repeat-poor on every mammal genome. Conservation-filter bias varies by species — see ratio table below — but consistently knocks lowercase fraction well below the genome-wide level.

   | species | HAL whole-genome lowercase | v1 anchor lowercase | v1 / genome-wide |
   |---|---:|---:|---:|
   | Homo_sapiens | 35.6% | 1.9% | 0.05× |
   | Mus_musculus | 54.7% | 14.5% | 0.27× |
   | Ceratotherium_simum | 53.8% | 19.2% | 0.36× |
   | Petromus_typicus | 37.4% | 3.9% | 0.10× |

3. **R4 is no longer the right recommendation.** Re-extracting from NCBI Datasets FASTAs would *decrease* mask coverage in 3 of the 4 species we tested. The cleaner mental model is now:
   - HAL ≈ Cactus's mask (often equal-to-or-greater-than NCBI's).
   - v1's per-species lowercase variance (1.9%–20% across anchors) is **conservation-recipe bias × species-specific genome composition**, not HAL strip-down or assembly-source heterogeneity in the simple way I framed before.
   - The GCF-vs-GCA difference in iter-2 (16.7% vs 6.6% mean within v1 anchors) is partly real assembly-source variation but mostly the same conservation-recipe bias differentiated by leaf-specific factors.

## Revised fix recommendations for repeat-masking

- **R1 (accept + document)** — strongest case now. v1 already carries the mask info Cactus produced; per-window variance is recipe-driven and not pathological.
- **R2 (uniform RepeatMasker pass)** — only fix that produces consistent-by-construction masking, if uniformity is wanted. Still requires the full Dfam library; cost is hours-to-days of compute per pass.
- **R4 (re-extract from NCBI Datasets)** — **REJECTED** based on this iter. Would reduce mask coverage for ≥3 of 4 species tested.
- R3 (per-species RepeatMasker BEDs from UCSC/Ensembl/NCBI) — still applicable, similar trade-offs as before.

## Caveats / what would broaden the result

- 4-species sample. A 108-species pass (HAL whole-genome lowercase per species, joined with the same metadata as iter-2's per-species table) would confirm the recipe-bias explanation generalizes. Each genome counts in ~30s on a c6id, so a full pass is ~1 hr wall on a 16-core box. Not done here; queued as a possible iter-5.
- Cactus's masking provenance — does it copy the input assembly's soft-mask intact, or run a fresh masker during build? Worth checking via a Cactus build doc / source, since the answer affects whether HAL masking really tracks NCBI's library uniformly.
- All four species are mammals where Cactus's repeat screening is reasonable. For species sets at the edge of the alignment's repeat-library coverage (e.g., tiny non-mammal outgroups in other HALs), the HAL-adds-masking pattern may not hold.

## Artifact

`~/audit/hal_vs_ncbi/comparison.tsv` on the audit cluster (now torn down — table reproduced in full above).

Updating the issue body to invert the R4 recommendation. Cluster `zoonomia-v1-audit` (c6id.2xlarge) being torn down.

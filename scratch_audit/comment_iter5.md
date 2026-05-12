🤖 **Iter 5 — Training-weight impact under 100× downweighting**

Hugely important context from the user: the training pipeline tokenizes case-insensitively but **downweights repeat (lowercase) bases by 100×** during loss/grad accumulation. So unlike I assumed in iter-4, the masking variance directly biases per-species effective training contribution. R1 (accept + document) is no longer free.

## Per-base effective training weight, given the 100× downweight

Define: `effective_weight_per_base = (1 - lower_frac - n_frac) · 1.0 + lower_frac · 0.01 + n_frac · 0.0` (treating N's as non-contributing). Aggregated by per-species totals from iter-2.

| species | acc | prefix | lower% | N% | upper% | **effective weight** |
|---|---|---|---:|---:|---:|---:|
| Lycaon_pictus | GCA_001887905.1 | GCA | 19.1% | 3.03% | 77.8% | **0.7803** |
| Ursus_maritimus | GCF_000687225.1 | GCF | 19.9% | 0.87% | 79.3% | 0.7946 |
| Ailurus_fulgens | GCA_002007465.1 | GCA | 19.8% | 0.44% | 79.8% | 0.7997 |
| Ceratotherium_simum | GCF_000283155.1 | GCF | 19.1% | 0.27% | 80.6% | 0.8081 |
| Vicugna_pacos | GCA_000767525.1 | GCA | 18.7% | 0.38% | 80.9% | 0.8111 |
| Equus_caballus | GCF_000002305.2 | GCF | 18.6% | 0.38% | 81.0% | 0.8121 |
| … | | | | | | |
| Petromus_typicus | GCA_004026965.1 | GCA | 3.9% | 0.01% | 96.1% | 0.9611 |
| Homo_sapiens | GCA_000001405.27 | GCA | 1.9% | 0.00% | 98.1% | **0.9813** |

**Per-species effective weight ranges 0.78 → 0.98 (1.26×).** Mean 0.91.

By accession source:

| accession prefix | n species | mean effective weight |
|---|---:|---:|
| **GCF** | 29 | **0.8319** |
| **GCA** | 79 | **0.9342** |

**GCA leaves get 10 pp more effective training weight per base** than GCF leaves. The direction is unintentional and is exactly what we'd expect from under-masking: assemblies whose submitters skipped or skimped on RepeatMasker carry repeats that don't get downweighted, so their windows contribute more gradient than they would with consistent masking. **The "less consistent than RefSeq" observation has a concrete training-bias signature: lower-quality (GCA) assemblies effectively over-contribute, lower-quality assemblies are the majority (73%) of v1 leaves.**

## Re-ranked fix options

R1 (accept) is no longer free. R2 / R3 become more compelling.

| ID | option | comments |
|---|---|---|
| **R2** *(now preferred if you care about the asymmetry)* | Run RepeatMasker uniformly on the 108 species genomes (mammal library); update each `{species}.2bit` with the unified mask; re-run `extract_sequences`. Eliminates the GCF/GCA gap by construction. | Cost: 108 genomes × ~1 hr each on 1 CPU ≈ 900 CPU-hours; on a c6id.12xlarge (48 cores) ~19 h wall, ~$50 EC2. Full Dfam library (~80 GB) download is a one-off setup cost. Cactus's own masking is *added on top* (iter-4) so this could *increase* per-window lowercase further — worth a dry run on a couple of species to gauge before committing. |
| **R1** | Document the bias; if downstream consumers know it's there, they can either compensate at the data-loader level (per-species weighting) or accept. | No code cost; carries a 10 pp / 1.26× training-weight asymmetry. |
| R3 | Per-species RepeatMasker BEDs from UCSC/Ensembl/NCBI (no fresh run). | Heterogeneous track availability across 108 leaves; quality-of-mask still varies by leaf. Won't close the GCF/GCA gap if NCBI's per-leaf mask is itself the issue. |
| ~~R4~~ | ~~Re-extract from NCBI Datasets FASTAs.~~ Rejected ([iter-4](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434656379)) — would *decrease* mask coverage in ≥3 of 4 species tested. | |

**Recommendation: R2 if uniform per-species training weight matters; R1 if a 10 pp per-prefix bias is acceptable** (it's small relative to other sources of training noise, and a 100×-downweighted base still contributes 1% of the gradient — most of the lift is on the 80–98% of upper-case bases that aren't downweighted at all).

A middle path worth considering: **R2 on the lowest-effective-weight leaves only** — re-mask the top-N most-under-masked GCA species (e.g., those at <10% lowercase, which is ~50 of 108 leaves), leaving the well-masked GCF leaves alone. Closes most of the gap at half the compute.

## What's still unknown

- How much does the 10 pp prefix gap actually move the model? Empirically: train two versions and compare on downstream evals. The audit can't answer that.
- Does Cactus's "extra" mask (iter-4 showed +18-24 pp for Ceratotherium/Mus over NCBI) come from a uniform Cactus-side repeat-detection pass, or from species-specific tracks fed in at build time? If uniform, then v1 is closer to R2-equivalent than it looks at the per-window level — the conservation-filter recipe is doing most of the variance, not assembly heterogeneity.
- Recipe interaction: ultraconserved-element-rich anchors (very low lowercase, very high effective weight) bias the model toward learning gene/regulatory regions. That's intentional. The R2 fix doesn't change that; it just normalises *across species* within the recipe.

Updating the issue body and the fix-option ranking.

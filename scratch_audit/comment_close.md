🤖 **Iter 3 — Closing summary + decision points**

Issue body updated with **Current findings** populated from iters 1+2 and **Fix options** ranked (F1 for N's, R4 for masking, with R1 as the "do nothing meaningful" alternative).

Net audit takeaways:

- **Neither anomaly is a code bug.** N-stretches are absent-species-side-filter; masking variance is upstream-driven (GCA assemblies under-masked at submission).
- **99.5% of v1 rows are N-free.** A 5% / 10% N-fraction threshold loses ≤0.5% of rows.
- **Per-assembly masking spread is ~10×** (1.9% → 20.0% lowercase). The GCF/GCA split alone explains most of it: GCF mean 16.7% ≈ RefSeq baseline 14.1%, GCA mean 6.6%.
- **`Homo_sapiens` in v1 is itself 1.9% masked** even though the upstream Ensembl `dna_sm` FASTA is ~50% masked — confirms the HAL → `hal2fasta` path drops most of the input mask info.

Decision points for the user:

1. **Does the training pipeline care about case?** If `a` and `A` are tokenized identically downstream, the masking issue is moot and **R1 (document + move on)** is the right answer. If not, **R4 (re-extract from per-assembly soft-masked NCBI FASTAs)** is the cleanest fix.
2. **Acceptable N-frac threshold for F1?** 5% loses ~0.5% of rows; 50% loses ~0.1%.

Cluster `zoonomia-v1-audit` (c6id.12xlarge in us-east-2) being torn down. Per-species stats parquet + the marginals TSVs persist on the audit cluster's `~/audit/section_bc/` and can be re-extracted from `s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/projection/min0.20/all_species_with_sequence.parquet` directly (~10 min on a c6id.12xlarge).

This issue is paused awaiting the case-sensitivity and F1-threshold calls before any follow-up PR.

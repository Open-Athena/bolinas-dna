# Evaluation Datasets Pipeline

This pipeline curates two variant-classification benchmark datasets from primary
sources, gene-matches negatives 1:1 against positives, and uploads the resulting
parquet datasets (plus their eval-harness sequence variants) to HuggingFace.

The curation is a from-scratch reimplementation of the TraitGym pipeline
(Benegas et al. 2025; songlab-cal/TraitGym) at the parent of the HGMD-removal
commit `e59d612e9`, so HGMD pathogenic SNVs are included as a positive source.

## Datasets

| Name | Description | Positives | Negatives |
|---|---|---|---|
| `mendelian_traits` | Mendelian disease pathogenic SNVs | HGMD ∪ OMIM ∪ Smedley et al. 2016 (de-duped, AF<0.001) | gnomAD common (AN≥25k, AF>0.05) |
| `complex_traits` | UKBB fine-mapped complex-trait variants | SuSiE+FINEMAP PIP > 0.9 across 119 traits | PIP < 0.01 (and not null in any trait) |

Each dataset has a corresponding `*_harness_255` eval-harness variant where a
255 bp window centered on each variant is materialized into
`context` / `ref_completion` / `alt_completion` columns. Models that prepend a
BOS token see 256 tokens of context; the rest of the codebase uses 255 bp
windows for the same reason.

## Matching scheme

For both datasets, every positive is matched 1:1 to one negative via TraitGym's
greedy-nearest-neighbor matcher (`bolinas.evals.matching.match_features`).
Matching is exact on every categorical feature, then Euclidean-nearest on the
(RobustScaler-scaled) continuous features, without replacement within a group.
See `src/bolinas/evals/matching.py` for the algorithm.

The matching design below was locked in [issue #156](https://github.com/Open-Athena/bolinas-dna/issues/156)
after a 24-iteration sweep. Two refinements close window-edge / MAF leaks the
basic continuous-only matching exhibited:

- **Splice pre-filter**: variants with `consequence_group == "splicing" AND
  exon_dist > 30` are dropped before matching (handful of mendelian splicing
  positives at non-coding-transcript splice sites whose protein-coding-only
  `exon_dist` is misleading; small loss in complex too — see issue #156 for
  per-subset retention tables).
- **Subset-conditional distance bins** added as exact-match categoricals on top
  of the continuous features (so continuous matching becomes a within-bin
  tie-breaker): `tss_dist_bin` only meaningful for `tss_proximal`, "NA"
  otherwise; `exon_dist_bin` only meaningful for `splicing`, "NA" otherwise.
  Edges in `bolinas.evals.matching` (`TSS_DIST_BIN_EDGES`,
  `EXON_DIST_BIN_EDGES`).

Per-dataset specifics:

- **mendelian_traits**:
  - continuous = `[tss_dist, exon_dist]`
  - categorical = `[chrom, consequence_final, tss_closest_gene_id,
    exon_closest_gene_id, tss_dist_bin, exon_dist_bin]`
- **complex_traits**:
  - continuous = `[tss_dist, exon_dist, MAF]` (`ld_score` is dropped from
    matching but kept as a passthrough column on the output dataset)
  - categorical = same as mendelian, plus an always-on `MAF_bin` (right-closed,
    log-spaced toward low MAF; `MAF_BIN_EDGES` in `bolinas.evals.matching`)

## Pipeline structure

`workflow/rules/common.smk` consolidates all shared infrastructure:

- `download_genome` — fetches the reference FASTA.
- `split_dataset_by_chrom` — generic chrom-based train/test split. Reads
  `results/dataset_unsplit/{dataset}.parquet` and emits both
  `results/dataset/{dataset}/train.parquet` and `.../test.parquet`. Train uses
  odd chromosomes (1, 3, …, X), test uses even (2, 4, …, Y).
- `materialize_eval_harness_dataset` — materializes the `_harness_{window_size}`
  variant of any dataset (`{base}_harness_{window_size}` naming convention).
- `hf_upload` — uploads `results/dataset/{dataset}/{train,test}.parquet` to
  `f"{output_hf_prefix}_{dataset}"` on HuggingFace.

`workflow/rules/intervals.smk`, `consequence.smk`, `gnomad_common.smk`,
`hgmd.smk`, `omim.smk`, `smedley.smk` build the shared per-source files
(GTF-derived TSS/exon intervals, per-chrom Ensembl VEP consequences, gnomAD
common variants, and the three Mendelian positive sources).

`workflow/rules/mendelian_traits.smk` produces:

```
positives.parquet              (HGMD+OMIM+Smedley, deduped, AF<0.001, consequences attached)
dataset_all.parquet            (positives ∪ gnomAD common, with build_dataset annotations)
dataset_unsplit/mendelian_traits.parquet  (gene-matched 1:1)
```

`workflow/rules/ldscore.smk` + `workflow/rules/complex_traits.smk` produce the
complex-trait dataset along the same lines, plus per-trait fine-mapping
downloads and aggregation across 119 traits.

The generic `split_dataset_by_chrom` rule then turns each
`results/dataset_unsplit/{name}.parquet` into the train/test pair, and
`hf_upload` ships them.

## Setup

Python dependencies are managed by the main project (see `../../README.md`).

Authenticate with HuggingFace before uploading:

```bash
huggingface-cli login
```

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/evals/`). A
default Snakemake profile at `workflow/profiles/default/config.yaml` configures
S3 storage and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: attach an IAM role with `AmazonS3FullAccess` to the instance.
- **On your laptop**: run `aws configure` with an IAM user's access key.

### Singularity (LD score)

`ldscore.smk` runs Hail inside a Docker image via Singularity to convert the
UKBB LD score HailTable to TSV. The host must have Singularity available and
be able to pull `hailgenetics/hail:0.2.130.post1-py3.11`. If Singularity isn't
available, the rule will fail; you can hand-build the LD-score parquet
elsewhere and stage it into `results/ldscore/UKBB.EUR.ldscore.parquet`.

### HGMD redistribution

HGMD pathogenic SNVs are downloaded from `sei-files.s3.amazonaws.com`. Including
HGMD-derived variants in a public HF dataset has license implications — check
before pushing the upload.

## Configuration (`config/config.yaml`)

Top-level keys:

| Key | Purpose |
|---|---|
| `genome_url` | Reference FASTA URL (GRCh38). |
| `annotation_url` | Ensembl GTF URL (release 107). |
| `consequences_repo` | HF repo with per-chrom Ensembl VEP consequences (`{chrom}.parquet`). |
| `gnomad_full_repo` | HF repo for the full gnomAD test parquet. |
| `gnomad_min_AN`, `gnomad_common_min_AF` | Filter for "common" gnomAD variants. |
| `tss_proximal_dist`, `exon_proximal_dist` | Distance thresholds for the `consequence_final` overrides. |
| `exclude_consequences` | High-impact VEP consequences dropped from the dataset. |
| `consequence_groups` | Mapping from collapsed-group key (`splicing`, `distal`) to the VEP consequences merged into that group. Single-consequence categories (`missense_variant`, `tss_proximal`, etc.) are not listed — they pass through unchanged. |
| `consequence_group_order` | Display name + plot order for each value that ends up in the `consequence_group` column. |
| `output_hf_prefix` | HF repo prefix; final repo is `f"{prefix}_{dataset}"`. |
| `datasets` | Which datasets `rule all` builds + uploads. |
| `mendelian_traits.*` | HGMD URL, Smedley URL, ClinVar release pin, submission summary date, AF threshold. |
| `complex_traits.*` | Fine-mapping repo, LD-score S3 path, PIP thresholds. |

`config/complex_traits.csv` lists the 119 UKBB traits used for `complex_traits`.

## Usage

```bash
# Edit configuration: snakemake/evals/config/config.yaml
uv run snakemake --directory snakemake/evals
```

To build a single dataset locally without uploading:

```bash
uv run snakemake --directory snakemake/evals \
  results/dataset/mendelian_traits/train.parquet \
  results/dataset/mendelian_traits/test.parquet
```

## Output

Datasets are uploaded to HuggingFace Hub at `f"{output_hf_prefix}_{dataset}"`.

Examples:

- `bolinas-dna/evals_mendelian_traits`
- `bolinas-dna/evals_complex_traits`
- `bolinas-dna/evals_mendelian_traits_harness_255`
- `bolinas-dna/evals_complex_traits_harness_255`

Locally, files live in `results/dataset/{dataset}/{train,test}.parquet`.

### Eval-harness columns

Datasets materialized with `_harness_{window_size}` add:

| Column | Description |
|---|---|
| `context` | Left flank up to (but not including) the variant position. |
| `ref_completion` | Reference allele + right flank. |
| `alt_completion` | Alternate allele + right flank. |
| `target` | Binary label (renamed from `label`). |

The window is centered on the variant: `context` has length `window_size // 2`
and each completion has length `window_size - window_size // 2`. For odd window
sizes, the extra base goes to the completion (right side).

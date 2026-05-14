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
| `mendelian_traits` | Mendelian disease pathogenic SNVs | HGMD ∪ OMIM ∪ Smedley et al. 2016 (de-duped, AF<0.001) | gnomAD common (AN≥25k, AF>0.001) |
| `complex_traits` | UKBB fine-mapped complex-trait variants | SuSiE+FINEMAP `max(PIP across the traits where this variant was fine-mapped) > 0.9` | `max(PIP) < 0.01` AND no SuSiE/FINEMAP combine-step null PIP among those traits (`label_variants_by_pip(use_null_pip_guard=True)`) |
| `eqtl` | GTEx v8 fine-mapped eQTLs from [eQTL Catalogue r7](https://www.ebi.ac.uk/eqtl/) (study `QTS000015`, 49 tissues × `ge` quantification, pooled) | SuSiE `max(PIP across tested tissues) > 0.9` from `credible_sets.tsv.gz` | `max(PIP) < 0.01` — variant must appear in at least one tissue's nominal sumstats (`all.tsv.gz`) but never reach a strong credible set; covered by the 0-fill in `bolinas.evals.catalogue_parser.merge_cs_and_sumstats` (sentinel: tested-but-no-signal variant has `pip=0`, not null, so `pl.max()` reaches it). |

Each dataset has a corresponding `*_harness_255` eval-harness variant where a
255 bp window centered on each variant is materialized into
`context` / `ref_completion` / `alt_completion` columns. Models that prepend a
BOS token see 256 tokens of context; the rest of the codebase uses 255 bp
windows for the same reason.

## Matching scheme

For each dataset, every positive is matched 1:1 to one negative via TraitGym's
greedy-nearest-neighbor matcher (`bolinas.evals.matching.match_features`).
Matching is exact on every categorical feature, then Euclidean-nearest on the
(RobustScaler-scaled) continuous features, without replacement within a group.
See `src/bolinas/evals/matching.py` for the algorithm.

The matching design below was locked in [issue #156](https://github.com/Open-Athena/bolinas-dna/issues/156)
at iter 33 after a 33-iteration sweep. Compared to the iter-22/24 production
design that this replaces, three things changed:

- **Per-biotype distances**: TSS / exon distances are computed twice, once
  against protein-coding transcripts (`distance_*_pc`) and once against all
  other transcripts (`distance_*_nc`, mostly lncRNAs / miRNAs / snoRNAs).
  Both pc and nc columns enter the matching as continuous features, and bin
  variants of them as categorical features (see below). The same applies to
  `*_closest_*_gene_id` keys.
- **Subset-conditional distance bins** as exact-match categoricals on top of
  the continuous features (so continuous matching becomes a within-bin
  tie-breaker):
  - `distance_tss_pc_bin` and `distance_tss_nc_bin` are meaningful only for
    `tss_proximal` (`"NA"` otherwise), edges `[0, 50, 100, 200, 500, 1000]`
    in both.
  - `distance_exon_pc_bin` is meaningful only for `splicing` (`"NA"`
    otherwise), edges `[0, 5, 20, 30]`.
  - `distance_exon_nc_bin` was tested and dropped — `splicing/dist_exon_nc`
    was already clean in the no-bin baseline.
- **Per-subset MAF binning** (complex_traits, eqtl): the iter-24 uniform 20bin
  scheme was replaced by a per-`consequence_group` tier (`MAF_TIERED_V1`):
  20bin for {distal, tss_proximal, ncRNA}, 10bin for {3'UTR, 5'UTR, missense},
  5bin for {synonymous, splicing, …small subsets}. For `eqtl/distal`
  specifically, fixed global edges leave an asymptotic Bonferroni-significant
  residual leak; replacing it with per-categorical-group **local equal-width
  log10(MAF) bins** (8 buckets, joint pos+neg ref over the categorical match
  key) closes the leak. That scheme is `MAF_TIERED_LOG8_DISTAL_ONLY`
  (= tiered_v1 with `LOG_LOCAL` for distal). Apply via
  `add_tiered_maf_bin(df, scheme, ...)`.

Per-dataset matching retention (n_pos / matched / %) is tracked in [issue #156](https://github.com/Open-Athena/bolinas-dna/issues/156).

Per-dataset specifics:

- **mendelian_traits** (no MAF column in the dataset_all parquet):
  - continuous = `[distance_tss_pc, distance_tss_nc, distance_exon_pc, distance_exon_nc]`
  - categorical = `[chrom, consequence_final, tss_closest_pc_gene_id,
    tss_closest_nc_gene_id, exon_closest_pc_gene_id, exon_closest_nc_gene_id,
    distance_tss_pc_bin, distance_tss_nc_bin, distance_exon_pc_bin]`
- **complex_traits**:
  - continuous = mendelian's + `MAF` (`ld_score` is dropped from matching but
    kept as a passthrough column on the output)
  - categorical = mendelian's + `MAF_bin` (per-subset, `MAF_TIERED_V1`)
- **eqtl**:
  - same continuous as complex_traits (the cohort-matched MAF comes natively
    from the eQTL Catalogue nominal sumstats, not joined from gnomAD)
  - same categorical as complex_traits but `MAF_bin` uses
    `MAF_TIERED_LOG8_DISTAL_ONLY` (tiered_v1 + per-group log_local for distal)
  - `tissues` is dropped from matching but kept as a passthrough column

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

`workflow/rules/eqtl.smk` produces the `eqtl` dataset from the
[eQTL Catalogue r7](https://www.ebi.ac.uk/eqtl/) per-tissue SuSiE
fine-mapping files. Source: study `QTS000015` (GTEx v8), all 49 tissue
datasets × the `ge` (gene-expression) quantification. Per tissue we
download two files from public FTP:

- `credible_sets.tsv.gz` (~4 MB) — PIP per credible-set member.
- `all.tsv.gz` (~3.5 GB) — full nominal sumstats; one row per (variant,
  gene) tested. Provides every variant tested in fine-mapping, including
  those that never reached a credible set. This is what makes the eqtl
  negative pool comparable to complex_traits' UKBB full-sumstats coverage
  (Finucane's GCS single-file release pre-filtered out the bulk of
  no-signal variants, capping the pool at ~1.3M; Catalogue gives us
  >10M).

`src/bolinas/evals/catalogue_parser.py` parses both files per tissue. The
critical step is `merge_cs_and_sumstats`: left-join the CS PIPs onto the
sumstats variant inventory, **fill `pip=0` (not null) for variants outside
any credible set**. This sentinel is load-bearing — `pl.max()` skips nulls,
so without the fill the tested-but-no-signal variants would fall through to
`label=None` and be excluded instead of labeled negative.

Cross-tissue aggregation in `eqtl_aggregate_tissues` then concats all 49
per-tissue parquets and labels via
`bolinas.evals.labeling.label_variants_by_pip`:

- `max(PIP across tested tissues) > 0.9` → positive
- `max(PIP) < 0.01` → negative
- intermediate `max(PIP)` ∈ `[0.01, 0.9]` → variant excluded from the dataset

Don't row-filter the input to extreme PIPs before the group_by — that
would silently mislabel a variant with tissue PIPs `[0.001, 0.5]` as a
clean negative (`max = 0.001` after dropping the 0.5 row) when its true
`max = 0.5` is intermediate and should exclude it. The labeling helper
has a unit test pinning down this regression
(`tests/evals/test_labeling.py:test_multiple_studies_max_excludes_when_intermediate`),
plus integration tests in `tests/evals/test_catalogue_parser.py` that
exercise the same case through the full Catalogue ingest path.

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

`ldscore_download` shells out to `aws s3 cp`, so install the `aws-cli` group
to get the `aws` binary on `uv run`'s PATH:

```bash
uv sync --group aws-cli
```

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
| `eqtl.*` | eQTL Catalogue study ID (`QTS000015` = GTEx v8), per-tissue list CSV, PIP thresholds. |

`config/complex_traits.csv` lists the 119 UKBB traits used for `complex_traits`.
`config/gtex_tissues.csv` lists the 49 GTEx v8 datasets used for `eqtl` (one
row per `(dataset_id, tissue_label, sample_size)` triple).

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
- `bolinas-dna/evals_eqtl`
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

# Dataset Creation Pipeline

This pipeline takes a list of genomes and creates training datasets by extracting genomic intervals, creating sliding windows, and uploading to HuggingFace.

## What it does

1. **Download genomes** - Downloads genome assemblies and GTF annotations from NCBI
2. **Extract intervals** - Identifies genomic regions based on recipe:
   - v1: Promoters (256bp upstream + 256bp downstream of TSS)
   - v2: mRNA exons + promoters (with flanking regions)
   - v3: CDS regions only
3. **Create windows** - Generates sliding windows over extracted regions
4. **Extract sequences** - Retrieves DNA sequences for each window
5. **Split data** - Creates train/validation splits by chromosome
6. **Merge and shard** - Combines data from multiple genomes and splits into shards
7. **Upload** - Uploads to HuggingFace Hub

## Genomic Region Extraction

The pipeline uses functions from `bolinas.data.utils` to extract genomic regions. These functions handle cross-species annotation differences (e.g., `transcript_biotype` vs `gbkey` attributes).

### Available Region Types

| Function | Description | Returns |
|----------|-------------|---------|
| `get_cds(ann)` | Coding sequences | GenomicSet |
| `get_promoters(ann, up, down, mRNA_only)` | Promoter regions around TSS | GenomicSet |
| `get_5_prime_utr(ann)` | 5' untranslated regions | GenomicSet |
| `get_3_prime_utr(ann)` | 3' untranslated regions | GenomicSet |
| `get_ncrna_exons(ann, biotypes)` | Functional ncRNA exons | GenomicSet |
| `get_mrna_exons(ann)` | mRNA exons with transcript_id | pl.DataFrame |

### ncRNA Filtering Criteria

The `get_ncrna_exons` function includes functional ncRNAs while excluding non-functional annotations:

**Included biotypes:** lnc_RNA, miRNA, snoRNA, tRNA, snRNA, rRNA, antisense_RNA, ncRNA, scRNA, vault_RNA, Y_RNA, scaRNA, RNase_P_RNA, RNase_MRP_RNA, telomerase_RNA, SRP_RNA, piRNA

**Excluded:**
- `pseudo "true"` attribute - annotated pseudogenes (~38k in human)
- `partial "true"` attribute - incomplete annotations (~24k in human)
- `transcript_biotype` containing "pseudogenic" (e.g., pseudogenic_tRNA in C. elegans)
- `gene_biotype` containing "pseudogene" (e.g., tRNA_pseudogene)
- `transcript_biotype == "transcript"` - used for transcribed pseudogenes in human
- `transcript_biotype == "primary_transcript"` - precursor RNAs before processing
- `description` or `product` containing "NMD candidate" - nonsense-mediated decay targets (~104 in human)
- `product` containing "LOW QUALITY" - explicitly flagged low quality (~43 in human)

### Promoter Options

`get_promoters(ann, n_upstream, n_downstream, mRNA_only=False)`:
- `mRNA_only=True`: Promoters from protein-coding mRNA transcripts only
- `mRNA_only=False` (default): Promoters from mRNA + functional ncRNA transcripts

### Cross-Species Compatibility

Functions handle both annotation styles:
- `transcript_biotype` / `gene_biotype` (human, mouse, most species)
- `gbkey` (some NCBI genomes like C. elegans, Merops nubicus)

For CDS extraction, both `feature == "CDS"` and `gbkey == "CDS"` are checked to handle C. elegans-style annotations where gene names appear in the feature column.

## Setup

Python dependencies are managed by the main project (see `../../../README.md` for installation).

Conda must be available for bioinformatics CLI tools.

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Required Parameters

- **`genomes_path`** - Path to filtered genome list (parquet file from genome_selection pipeline)

- **`intervals`** - List of interval types to generate
  - Format: `{recipe}/{window_size}/{overlap}`
  - Examples:
    - `v1/512/256` - Promoters with 512bp windows and 256bp overlap
    - `v2/512/256` - mRNA + promoters with 512bp windows and 256bp overlap
    - `v3/512/256` - CDS regions with 512bp windows and 256bp overlap
  - Recipes:
    - **v1**: Promoters only (256bp upstream + 256bp downstream from TSS)
    - **v2**: mRNA exons + promoters
    - **v3**: CDS regions only

- **`genome_sets`** - Taxonomic groupings for separate datasets
  - Each set defined by `name`, `rank_key`, and `rank_value`
  - Example: `{name: "mammals", rank_key: "class", rank_value: "Mammalia"}`
  - Creates separate datasets for each taxonomic group

- **`output_hf_prefix`** - HuggingFace repository prefix (e.g., "username/dataset-name")

### Optional Parameters

- **`validation_chroms`** - List of chromosome accessions for validation split
  - Default: Uses specific chromosomes for train/val split
  - Example: `["NC_000019.10"]` (human chr19)

- **`shuffle_seed`** - Random seed for reproducibility (default: 42)

- **`n_shards`** - Number of shards to split dataset into (default: 64)
  - More shards = better parallelization

- **`add_rc`** - Add reverse complement sequences (default: true)
  - Doubles dataset size by including reverse complements

## Usage

```bash
# Ensure you have a genome list from the genome_selection pipeline
# Copy or link it to config/genomes.parquet

# Edit configuration file: config/config.yaml

# Run pipeline
uv run snakemake --cores all --use-conda
```

## Output

Datasets are uploaded to HuggingFace Hub at the specified `output_hf_prefix`.

Dataset naming format: `{output_hf_prefix}-{genome_set}-{recipe}-{window_size}-{overlap}`

Example: `username/genomes-v3-mammals-v2-512-256`

Local intermediate files are stored in `results/` (not committed to git).

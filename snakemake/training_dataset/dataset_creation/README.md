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

### UTR Extraction

The `get_5_prime_utr` and `get_3_prime_utr` functions compute UTR regions by finding exon portions outside the CDS boundaries for each transcript:

- **5' UTR**: Exon regions before CDS start (+ strand) or after CDS end (- strand)
- **3' UTR**: Exon regions after CDS end (+ strand) or before CDS start (- strand)

**CDS exclusion**: Regions that are CDS in *any* transcript are excluded from UTRs, even if they are UTR in another transcript. This handles alternative transcripts where the same genomic region may be coding in one isoform but non-coding in another.

**Verification identity**: By definition, mRNA exons consist of CDS + 5' UTR + 3' UTR, so:

```
mRNA - CDS = 5' UTR | 3' UTR
```

This identity can be used to verify the correctness of UTR extraction:

```python
mrna_exons = GenomicSet(get_mrna_exons(ann))
cds = get_cds(ann)
utr5 = get_5_prime_utr(ann)
utr3 = get_3_prime_utr(ann)

# These should be equal (or nearly equal)
mrna_minus_cds = mrna_exons - cds
utr_union = utr5 | utr3

# Check differences
diff1 = mrna_minus_cds - utr_union  # Should be ~0
diff2 = utr_union - mrna_minus_cds  # Should be 0
```

**Edge cases**: A small number of intervals (17 in human, ~13kb total) satisfy `mRNA - CDS ⊃ 5' UTR | 3' UTR` but not strict equality. These are regions in mRNA exons that are neither CDS nor UTR:

| chrom | start | end | notes |
|-------|-------|-----|-------|
| NC_000015.10 | 64691514 | 64691515 | 1bp gap within CDS |
| NC_000019.10 | 2271442 | 2271443 | 1bp gap within CDS |
| NT_167249.2 | 941944 | 942223 | unplaced scaffold |
| NT_187518.1 | 162112 | 162469 | unplaced scaffold |
| NT_187610.1 | 130379 | 130698 | unplaced scaffold |
| NT_187633.1 | 301994 | 302395 | unplaced scaffold |
| NT_187646.1 | 158490 | 158847 | unplaced scaffold |
| NW_003315955.1 | 78827 | 79107 | unplaced scaffold |
| NW_009646197.1 | 443494 | 451168 | unplaced scaffold |
| NW_009646203.1 | 106479 | 106757 | unplaced scaffold |
| NW_009646203.1 | 107409 | 107495 | unplaced scaffold |
| NW_018654714.1 | 1278 | 1611 | unplaced scaffold |
| NW_019805490.1 | 22254 | 22576 | unplaced scaffold |
| NW_019805490.1 | 22919 | 23010 | unplaced scaffold |
| NW_019805496.1 | 0 | 22 | unplaced scaffold |
| NW_025791794.1 | 0 | 1856 | unplaced scaffold |
| NW_025791802.1 | 234037 | 234878 | unplaced scaffold |

These are typically 1bp gaps within CDS (small introns or frameshift annotations) or unusual annotations on unplaced scaffolds.

### Annotation Sources and Data Quality

NCBI genome annotations come from different sources with varying quality levels:

| Source | Description | Transcript Prefix |
|--------|-------------|-------------------|
| **BestRefSeq** | Curated, high-confidence | NM_, NR_ |
| **Gnomon** | Computational predictions | XM_, XR_ |
| **RefSeq** | Reference sequences | NM_, NR_ |
| **cmsearch** | RNA structure-based | - |
| **tRNAscan-SE** | tRNA predictions | - |
| **Curated Genomic** | Manual curation | - |

**Cross-species distribution** (based on 34 genomes in `genome_subset_analysis`):

| Source | Mean | Median |
|--------|------|--------|
| Gnomon | 81% | 97% |
| RefSeq | 11% | 0% |
| BestRefSeq | 3% | 0% |
| cmsearch | 2% | 1% |
| tRNAscan-SE | 2% | 1% |

**Key findings:**
- **Gnomon dominates** most genomes (median 97% of transcripts)
- **BestRefSeq** is only significant in well-studied model organisms:
  - Human: 53.5%
  - Mouse: 43.9%
  - Zebrafish: 13.5%
  - Chicken: 12.3%

**Implications for training data quality:**
- Filtering to BestRefSeq-only would work for model organisms but eliminate most data for other species
- Gnomon predictions tend to have more extreme outliers (very long UTRs from alternative transcripts)
- Interval length outliers by region (human):

| Region | p99 | Max | >50kb |
|--------|-----|-----|-------|
| CDS | 1.3kb | 21kb | 0 |
| Promoters | 1.2kb | 3.5kb | 0 |
| 3' UTR | 9kb | 56kb | 1 |
| 5' UTR | 5kb | 88kb | 3 |
| ncRNA | 7kb | 92kb | 8 |

CDS regions have no extreme outliers (biological constraint on exon size), while UTRs and ncRNA exons can be very large. Consider applying length filters to UTR/ncRNA regions if outliers are problematic.

To compute annotation source statistics:
```bash
uv run snakemake all_annotation_source_stats --cores 4
```

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

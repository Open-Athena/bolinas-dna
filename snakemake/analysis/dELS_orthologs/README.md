# dELS_orthologs

Pipeline for evaluating how well **cheap pairwise sequence-similarity methods**
recover **hg38 ↔ mm10 orthologous dELS (distal enhancer-like signature) pairs**
that the **Zoonomia 241-way Cactus** multiple-genome alignment identified as
orthologs (gold standard:
[`hg38-mm10-Homologous.tsv`](https://downloads.wenglab.org/hg38-mm10-Homologous.tsv)).

Tracking issue: [Open-Athena/bolinas-dna#120](https://github.com/Open-Athena/bolinas-dna/issues/120).

## Hello-world scope: ZRS locus, asymmetric search

This iteration is intentionally tiny. It runs the entire pipeline (download,
class filter, repeat filter, align with each configured aligner, hit→cCRE
annotation) on the **ZRS limb-enhancer locus** with an asymmetric search
region:

- **Query** (hg38): dELS that fall *inside biological ZRS proper* —
  `chr7:156,790,115–156,793,672`, no flank — optionally narrowed further to a
  configured accession list (`query_accessions` in config; `null` = every
  dELS in the window).
- **Target** (mm10): one ~206 kb genomic interval covering ZRS ± 100 kb —
  `chr5:29,212,497–29,418,634` — flank gives the mouse ortholog room to sit
  anywhere within the wider locus.
- **Aligners** (configured via `aligners` in config):
  - **mmseqs2** — nucleotide search at `-s 7.5`, no identity filter,
    `--strand 2`, `--mask-lower-case 1`.
  - **minimap2** — `-x map-ont` preset (asm5/10/20 are too strict for
    ~20% divergent non-coding dELS at 200–400 bp; map-ont's noisy-read
    scoring and smaller k-mer seed both alignments).

Each aligner produces hits in a shared schema (`results/align/{aligner}/hits.tsv`)
and its own per-query report (`results/eval/{aligner}/per_query_report.tsv`).

The pipeline runs in seconds because the query set is tiny and the target is
small. It answers a focused question per query: *does each method recover the
same mouse ortholog the Cactus-derived gold standard identifies?*

## Run

```bash
cd snakemake/analysis/dELS_orthologs
uv run snakemake -n            # dry-run
uv run snakemake               # full run; takes a few minutes (genome 2bits dominate)
```

Conda envs (`workflow/envs/{mmseqs2,minimap2,bioinformatics}.yaml`) are
installed automatically via `--use-conda` from the profile.

## Filters

The class and repeat-mask filters apply **only to the hg38 query side**:

1. **Class** (`filter_cres_to_dels`, hg38): keep only Registry-V4 entries
   with `cre_class == "dELS"`. The mm10 candidate pool is intentionally
   class-agnostic so the per-query report can show hits that overlap mouse
   cCREs of *any* class — including mouse `CA` cCREs that are the gold-
   standard orthologs of human dELS in this locus (a known
   classification-asymmetry; see the tracking issue).
2. **Window** (`subset_dels_to_window`, hg38): restrict to dELS inside the
   hg38 search region.
3. **Accession list** (`select_query_accessions`, hg38): if
   `query_accessions` is set in config, restrict to those specific
   accessions. Default for v1 hello-world: `["EH38E2604086"]`.
4. **Repeat-mask** (`filter_query_by_repeat`, hg38): drop queries whose
   soft-masked (lowercase) fraction in the UCSC `_sm` 2bit exceeds
   `max_soft_masked_frac` (default `0.5`). With `--mask-lower-case 1`,
   mmseqs2 has no k-mer seeds in heavily-masked sequence.

mm10 only goes through `subset_mm10_cres_to_window` (any class, no repeat
filter) — that BED is used solely to annotate which mm10 cCRE each mmseqs hit
overlaps, not to seed the search.

## Outputs

Headline:

- `results/eval/{aligner}/per_query_report.tsv` — for each query, top-k hits
  from this aligner with absolute mm10 coordinates, the overlapping mm10
  cCRE accession + class + coords, and an `in_gold_standard` flag. Gold-
  standard partners that the aligner did **not** recover are appended as
  null-score rows; post-filter queries with no hits and no gold partner get
  a single null row so they remain visible. Mirror parquet alongside.
- `results/align/{aligner}/hits.tsv` — unified-schema per-aligner hits
  (`query, hit_chrom, hit_start, hit_end, rev_strand, score, fident,
  evalue, qcov, tcov`) in absolute mm10 BED coordinates.
- `results/sanity/self_recall.parquet` — hg38 query-vs-itself recall@1 via
  mmseqs2 (sanity floor; should be 1.0).

Intermediate:

- `results/cre/hg38/cres.parquet` → `dels.parquet` → `dels_window.parquet` →
  `query.parquet` → `query.filtered.parquet` (post repeat-mask).
- `results/cre/hg38/query.fasta`, `query.filtered.fasta` — per-dELS query sequences.
- `results/cre/mm10/cres.parquet` → `cres_window.parquet` (any class).
- `results/target/mm10_window.fasta` — single ~206 kb mm10 ZRS window FASTA.
- `results/orthologs/hg38_mm10.tsv` — gold-standard ortholog pairs (full file, ~280k rows).
- `results/search/hits.tsv` — raw mmseqs2 alignments (`query,target,tstart,tend,bits,evalue,fident,qcov,tcov`).

## Configuration

`config/config.yaml`:

- `cre_class` — class filter target on the hg38 query side (default `"dELS"`).
- `cre_urls`, `genome_urls`, `ortholog_url` — input data sources.
- `search_region.{hg38,mm10}` — biological coords + per-species `flank_bp`.
  hg38 uses `flank_bp: 0`, mm10 uses `flank_bp: 100000`.
- `query_accessions` — optional list of accessions to use as queries; comment
  out to use every dELS in the hg38 search region.
- `mmseqs2.{sensitivity,max_accept}` — search parameters.
- `max_soft_masked_frac` — repeat-mask filter threshold for the query side.
- `report_top_k` — how many top-bits hits to keep per query in the report.

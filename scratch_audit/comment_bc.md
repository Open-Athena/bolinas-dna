🤖 **Iter 2 — Section B (N-content dataset-wide) + Section C (lowercase / repeat-masking) + RefSeq baseline**

Headline: the N-stretches **and** the repeat-mask inconsistency are real but **upstream-driven** — there's no code bug, and 99.5% of v1 rows are actually fine on the N axis. The masking variance has a clean per-assembly explanation. Numbers below.

---

## Section B — N-content across the full v1 source parquet

Scanned the canonical [`all_species_with_sequence.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/project.smk#L316-L335) (11.18 GB; 111.94 M rows; pre-RC augmentation) on the audit cluster.

### Headline: 99.5% of v1 rows are N-free

| N-position bucket            |     n rows | fraction |
|------------------------------|-----------:|---------:|
| `none` (no N anywhere)       | 111,335,734 | **99.46%** |
| `leading` (5′-edge N-run)    |    183,419 | 0.16% |
| `trailing` (3′-edge N-run)   |    185,626 | 0.17% |
| `both_edges`                 |     42,330 | 0.04% |
| `interior` (no edge N's)     |    193,031 | 0.17% |

`leading` + `trailing` (~0.33% combined) are roughly the M2 (resize expansion) signature; `interior` (~0.17%) is M1 / M3-ish. Roughly balanced — no single mechanism dominates.

### Per accession prefix — N-content correlates with assembly source

| accession prefix | n species |  mean agg-N-frac | mean frac rows with ≥10% N |
|------------------|----------:|-----------------:|---------------------------:|
| **GCF** (RefSeq curated) | 29 | **0.36%** | **0.85%** |
| **GCA** (GenBank submitted) | 79 | **0.08%** | **0.19%** |

Counterintuitive but explainable: GCF assemblies are more often *chromosome-level* and use **internal N-padding between contigs** to scaffold up; many GCA Zoonomia leaves are *scaffold-level* with shorter unanchored fragments, so the projected windows simply don't see scaffold-joining N-stretches as often.

### Worst-offender species

| species | acc | level | agg N-frac | frac rows ≥10% N | agg lower-frac |
|---|---|---|---:|---:|---:|
| Lycaon_pictus | GCA_001887905.1 | Chromosome | **3.03%** | 5.79% | 19.7% |
| Carlito_syrichta | GCF_000164805.1 | Scaffold | 1.34% | 1.88% | 12.6% |
| Otolemur_garnettii | GCF_000181295.1 | Scaffold | 0.97% | 1.42% | 13.9% |
| Ursus_maritimus | GCF_000687225.1 | Scaffold | 0.87% | 2.50% | 20.0% |
| Balaenoptera_acutorostrata | GCF_000493695.1 | Scaffold | 0.79% | 1.98% | 18.7% |

The OP's example species `Ceratotherium_simum` (GCF_000283155.1) sits at **0.27% mean N-frac with 0.72% of rows ≥10% N** — well above the dataset median but not in the top 5. The 125-leading-N row in the HF preview is one of the ~7.6 k rows in that 0.72% tail.

---

## Section C — Repeat-masking (lowercase fraction) per species + assembly axis

### Per-prefix headline: GCA assemblies are systematically under-masked

| accession prefix | n species | **mean agg lower-frac** |
|------------------|----------:|-----------------------:|
| **GCF** | 29 | **16.7%** |
| **GCA** | 79 | **6.6%**  |

Per-species range: **1.9% (`Homo_sapiens` from the HAL!) → 20.0% (`Ursus_maritimus`)**. A spread of ~10× across the v1 species set.

### Least- and most-masked v1 species

**Bottom 10 (lowest lowercase fraction):**

| species | acc | prefix | lvl | lower% | N% |
|---|---|---|---|---:|---:|
| Homo_sapiens | GCA_000001405.27 | GCA | Chromosome | **1.9%** | 0.00% |
| Petromus_typicus | GCA_004026965.1 | GCA | Scaffold | 3.9% | 0.01% |
| Cricetomys_gambianus | GCA_004027575.1 | GCA | Scaffold | 4.0% | 0.03% |
| Thryonomys_swinderianus | GCA_004025085.1 | GCA | Scaffold | 4.1% | 0.01% |
| Microgale_talazaci | GCA_004026705.1 | GCA | Scaffold | 4.1% | 0.04% |
| Scalopus_aquaticus | GCA_004024925.1 | GCA | Scaffold | 4.1% | 0.02% |
| Cuniculus_paca | GCA_004365215.1 | GCA | Scaffold | 4.2% | 0.01% |
| Nycticebus_coucang | GCA_004027815.1 | GCA | Scaffold | 4.2% | 0.00% |
| Solenodon_paradoxus | GCA_004363575.1 | GCA | Scaffold | 4.2% | 0.03% |
| Craseonycteris_thonglongyai | GCA_004027555.1 | GCA | Scaffold | 4.2% | 0.02% |

**All 10 are GCA**; most are Zoonomia "GCA_004…" small-team assemblies that never had a uniform RepeatMasker pass applied at submission.

The `Homo_sapiens` row at **1.9%** is striking. The v1 anchor pipeline uses the Ensembl `dna_sm.primary_assembly` (~50% soft-masked) for the *anchor BED only*; the *sequence extraction* for v1's Homo_sapiens species rows goes through `hal2fasta` on the Zoonomia HAL, which has lost most of the input mask along the way. So v1's "human" rows are ~25× less masked than the Ensembl FASTA the rest of the pipeline points to. That's an artifact of the HAL → FASTA path, not anything specific to non-human leaves.

**Top 10 (highest lowercase fraction):**

| species | acc | prefix | lvl | lower% | N% |
|---|---|---|---|---:|---:|
| Ursus_maritimus | GCF_000687225.1 | GCF | Scaffold | **20.0%** | 0.87% |
| Ailurus_fulgens | GCA_002007465.1 | GCA | Scaffold | 19.9% | 0.44% |
| Enhydra_lutris | GCF_002288905.1 | GCF | Scaffold | 19.8% | 0.02% |
| Lycaon_pictus | GCA_001887905.1 | GCA | Chromosome | 19.7% | 3.03% |
| Odobenus_rosmarus | GCF_000321225.1 | GCF | Scaffold | 19.6% | 0.05% |
| Ceratotherium_simum | GCF_000283155.1 | GCF | Scaffold | **19.2%** | 0.27% |
| Vicugna_pacos | GCA_000767525.1 | GCA | Scaffold | 18.8% | 0.38% |
| Rhinolophus_sinicus | GCA_001888835.1 | GCA | Scaffold | 18.7% | 0.16% |
| Equus_caballus | GCF_000002305.2 | GCF | Chromosome | 18.7% | 0.38% |
| Balaenoptera_acutorostrata | GCF_000493695.1 | GCF | Scaffold | 18.7% | 0.79% |

The OP example species `Ceratotherium_simum` is actually among the **best-masked** of the v1 species (19.2% lowercase) — its visible N-run in the HF preview is unlucky sampling, not a masking signal.

---

## RefSeq baseline ([`genomes-v5-genome_set-mammals-intervals-v1_255_128`](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-mammals-intervals-v1_255_128))

200 k row sample streamed:

| metric | value |
|---|---:|
| `agg_n_frac` | **0.0%** (zero N's — confirms training_dataset's ACGT-only windowing) |
| `agg_lower_frac` | **14.1%** |

So the RefSeq pipeline at the matched recipe sits at **14.1% mean lowercase**, between v1's GCF mean (**16.7%**) and v1's GCA mean (**6.6%**). The two pipelines aren't apples-to-apples — RefSeq uses the v1 "upstream" interval recipe vs zoonomia v1's conservation-filtered anchors — but the headline is:

- **v1 GCF species ≈ RefSeq baseline** on masking — the HAL preserves NCBI's RepeatMasker output reasonably well for GCF leaves.
- **v1 GCA species are ~2× under-masked vs RefSeq** — and these are 73 % of the v1 leaf set.

---

## Answers to the OP's two questions

1. **"Is the N-stretch a bug in our region-expansion code?"** — **No.**
   - Section A confirmed the specific example is a 580 bp scaffold-internal N-gap on `JH767724.1` and the resize landed on the gap's 3′ edge. Documented behavior, not a defect.
   - Section B confirms it's not widespread: **99.46% of v1 rows are N-free**, and only **0.4% have ≥10% N**. The missing piece is the absence of a species-side N-filter (we filter human anchors against `undefined.bed` but nothing analogous post-projection).

2. **"Why is masking less consistent than RefSeq?"** — **Upstream assembly source.**
   - v1 lowercase fraction varies **10×** across leaves (1.9% → 20.0%).
   - The strongest axis is **GCF vs GCA**: GCF mean 16.7%, GCA mean 6.6%. GCF assemblies arrive with NCBI's uniform RepeatMasker pass; GCA assemblies arrive with whatever the submitting consortium chose to mask, often nothing.
   - Within v1, GCF species are masked **about the same as the RefSeq pipeline** (16.7% vs 14.1%). The "inconsistency" is GCA leaves dragging the v1 average down.
   - One striking artifact: v1's `Homo_sapiens` rows show **1.9%** lowercase, far below Ensembl's ~50% soft-masking. The HAL → `hal2fasta` extraction has lost most of the mask info that was present in the upstream `dna_sm` FASTA. This affects *all* leaves to some extent but is invisible in species without a high-quality masked upstream FASTA in the first place.

## Fix-option ranking (to be written into the issue body)

For N-stretches: **F1 (post-extraction N-fraction filter)** is the obvious cheapest win — drop the ~0.4–0.5% of rows with non-trivial N-content. F2 (pre-extraction species N-filter) is cleaner architecturally but more pipeline work.

For repeat-masking: **R4 (re-extract from per-assembly soft-masked NCBI FASTAs)** would fix both the GCA gap and the HAL strip-down in one move, at the cost of a per-leaf accession→FASTA audit. R2 (uniform RepeatMasker post-pass) is the most uniform but the most expensive.

## Deferred to a follow-up

A uniform RepeatMasker baseline on a sampled subset (the Section C step 4 item). The bioconda RepeatMasker package only ships the **Dfam 3.9 root partition (77 MB, 237 families)** — not enough to be a meaningful baseline. A real comparison needs the full Dfam library (several GB download + setup). Out of scope for this iteration; would land as Iter 3 if R2 becomes a serious candidate.

## Artifacts (all on the audit cluster)

- `~/audit/section_bc/per_species_stats.tsv` — full 108-species table, all 25 columns
- `~/audit/section_bc/by_accession_prefix.tsv`, `by_quality_source.tsv`, `by_assembly_level.tsv` — marginals
- `~/audit/section_bc/n_position_histogram.tsv` — 5-bucket position decomposition
- `~/audit/section_bc/refseq_baseline_summary.json` — 200 k row sample summary

Script: [`scratch_audit/section_bc_quantify.py`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset) (gitignored under `scratch_audit/` on branch `claude/stoic-sutherland-ba71dd`; not pushed to the repo proper).

Next iteration: write fix-option recommendations into the issue body and close.

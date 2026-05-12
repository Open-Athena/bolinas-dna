## Question

Two anomalies were observed in the public [`bolinas-dna/zoonomia-v1-v1`](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-v1) HuggingFace training dataset:

1. **N-stretches in training windows.** Example row from page 2 of v1:

   ```
   win_7_000231918  Ceratotherium_simum  JH767724.1  24,562,726  24,562,981  +  78,580,661
   NNNN…NNN (125 leading N's) CGCCGCCGCCGCG…
   ```

   A 255 bp window with ~125 N's at its 5′ end. First thing to rule out: a bug in our region-expansion code (`bolinas.projection.resize.resize_to_length`). Other plausible non-bug causes: real species assembly gap, contig-boundary effect, tiny target contig.

2. **Repeat-masking less consistent than RefSeq.** The prior RefSeq-based [`training_dataset/dataset_creation`](https://github.com/Open-Athena/bolinas-dna/tree/a38154132b173bffe0fb5d589682656b327759fd/snakemake/training_dataset/dataset_creation) pipeline produced sequences with reliable soft-masking. v1 looks much less uniformly masked. User read: upstream-driven — Zoonomia 447 Cactus alignment ingested per-leaf assemblies with varying RepeatMasker coverage.

## Scope

**In:** trace the specific example anchor; per-species N + lowercase stats over the full v1 source parquet joined with assembly metadata (GCF/GCA, quality_source, contig_n50, assembly_level); HAL-derived FASTA vs NCBI Datasets FASTA per-species lowercase comparison (iter 4). Fix options written into this body, no implementation.

**Out:** implementing fixes; re-running halLiftover on the full HAL to recover pre-resize spans; rebuilding the v1 dataset. RepeatMasker comparison **deferred** — the bioconda RepeatMasker ships with only the Dfam 3.9 root partition (77 MB, 237 families), insufficient for a meaningful baseline; needs full Dfam library (~GBs) which is out of scope for this iteration.

## Approach

Single SkyPilot **c6id.12xlarge** in `us-east-2`, tagged `project=dna` (initially `c6id.4xlarge` but Polars streaming OOM'd on the 11 GB source parquet; stepped up after one re-launch).

Branch [`claude/stoic-sutherland-ba71dd`](https://github.com/Open-Athena/bolinas-dna/tree/claude/stoic-sutherland-ba71dd); analysis SHA [`a381541`](https://github.com/Open-Athena/bolinas-dna/commit/a38154132b173bffe0fb5d589682656b327759fd) (= `main`). All scratch scripts under gitignored `scratch_audit/`.

Mechanisms-under-test for the N-stretches (we *distinguished* them, didn't commit up front):

- **(M1)** Lift inside a real assembly gap.
- **(M2)** Resize expansion into an adjacent gap.
- **(M3)** Contig-edge effects (tiny target contig + scaffold-context N-padding).
- **(M4)** Unanticipated.

## Outcome

**Neither finding is a big issue in practice.**

- N-stretches: 0.155% of bases are N; 99.46% of windows are N-free; 0.4% of windows have ≥10% N. Not a code bug — it's the absence of a species-side N-filter, with multiple mechanisms (lift-in-gap M1, resize-into-gap M2, contig-edge M3) contributing roughly evenly. The OP's "125-leading-N" example was unlucky sampling out of ~7600 such rhino rows.
- Repeat masking: overall **9.3% lowercase** across v1 → effective per-base training weight **0.907** under the 100× downweight policy. Per-species effective weight 0.78–0.98 (1.26× spread). The 10 pp GCF/GCA gap in v1 anchors is the only piece that's clearly *not* biology — it tracks Dfam 3.3 library coverage by lineage (which was uniform across leaves at the Zoonomia protocol level). Iter-4 + iter-5 confirmed the pipeline already does what a uniform-RepeatMasker pass would do: `RepeatMasker(Dfam 3.3, target 40-50% genome-wide) ∪ cactus-preprocess RED`.

**Recommendation: R1 + optional F1.** Document both findings in the dataset card; F1 is a near-free one-rule pipeline edit. Reopen if downstream evals surface a real training-side artifact from the per-species mask variance.

See the [closing comment](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434815057) for the practical bottom line and [iter-5](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434757618) for the masking-pipeline mechanism.

## Findings

**Both issues are real but upstream-driven. No code bugs. 99.5% of v1 rows are N-free; the masking variance has a clean per-assembly explanation.**

### Section A — example trace ([iter-1 comment](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4433767007))

- Human anchor `win_7_000231918` at `7:29685376-29685631` has **0 N's in hg38** — the [`windows.smk` filter](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/windows.smk#L32) did its job.
- The rhino window at `JH767724.1:24562726-24562981` reproduces from `Ceratotherium_simum.2bit` with 125 leading N's matching the HF row exactly.
- `twoBitInfo -nBed` shows **one 580 bp gap** at `JH767724.1:24562271-24562851`. The window's 5′ 125 bp sit inside the gap; the 3′ 130 bp are real sequence. **Lift midpoint maps 2 bp past the gap edge.** Resize expansion into a gap (M2) is the most plausible single-anchor explanation; M1 cannot be ruled out without re-running halLiftover.
- Not a code bug — the [`resize.py` docstring](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/src/bolinas/projection/resize.py#L1-L10) explicitly notes that padded bases beyond the orthologous extent are adjacent species genome.

### Section B — dataset-wide N quantification ([iter-2 comment](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434090865))

Scanned all 111.94 M v1 rows.

| N-position bucket  |     n rows | fraction |
|--------------------|-----------:|---------:|
| `none`             | 111,335,734 | **99.46%** |
| `leading` (M2 sig.)|    183,419 | 0.16% |
| `trailing` (M2 sig.)|   185,626 | 0.17% |
| `both_edges`       |     42,330 | 0.04% |
| `interior` (M1/M3) |    193,031 | 0.17% |

Roughly balanced between edge and interior — no single mechanism dominates. Worst-offender species: `Lycaon_pictus` (GCA_001887905.1) at **3.03% mean N-frac**; the OP species `Ceratotherium_simum` (GCF_000283155.1) sits at 0.27%.

| accession prefix | n species |  mean agg-N-frac | mean frac rows ≥10% N |
|------------------|----------:|-----------------:|----------------------:|
| **GCF** | 29 | 0.36% | 0.85% |
| **GCA** | 79 | 0.08% | 0.19% |

Chromosome-level GCF assemblies pad inter-contig gaps with N's; scaffold-level GCA fragments often don't.

### Section C — repeat-masking + RefSeq baseline ([iter-2 comment](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434090865))

| accession prefix | n species | **mean agg lower-frac** |
|------------------|----------:|------------------------:|
| **GCF** | 29 | **16.7%** |
| **GCA** | 79 | **6.6%** |

Range: **1.9% (`Homo_sapiens` from the HAL!) → 20.0% (`Ursus_maritimus`)** — a **~10× spread** across leaves.

No fair single-number RefSeq-pipeline baseline exists — every recipe in the prior pipeline is a functional-region selection (TSS-centered, CDS, downstream, enhancer) with its own repeat-content bias, and none is composition-equivalent to v1 zoonomia's conservation-filtered cross-mammal anchors (see [scope correction](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434164215)). The masking story is mostly a recipe-bias story:

- **HAL preserves mask info — it doesn't strip it.** Direct whole-genome comparison ([iter-4 comment](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434668199)) on four representative species:

  | species | accession | HAL whole-genome | NCBI Datasets |
  |---|---|---:|---:|
  | Homo_sapiens | GCA_000001405.27 | 35.6% | 38.0% |
  | Ceratotherium_simum | GCF_000283155.1 | **53.8%** | 29.6% |
  | Petromus_typicus | GCA_004026965.1 | 37.4% | 34.0% |
  | Mus_musculus | GCF_000001635.26 | **54.7%** | 36.4% |

  HAL ≥ NCBI Datasets in all four cases; sometimes substantially so (Ceratotherium 1.82×, Mus 1.50×). Plausible: Cactus applies its own repeat-screening during alignment construction.

- **The "Homo_sapiens v1 at 1.9%" finding was a recipe artifact, not a HAL strip-down.** v1's anchors are conservation-filtered (phyloP_447m `proportion_conserved ≥ 0.20`), selecting overwhelmingly CDS / ultraconserved / regulatory regions — intrinsically repeat-poor. Whole-genome HAL Homo_sapiens is **35.6%** lowercase; the v1 anchor set is **1.9%** — a 19× cut driven by the recipe, not the source FASTA.

- The conservation-filter bias varies by species (`v1 anchor lowercase / HAL whole-genome lowercase`): Homo_sapiens 0.05×, Petromus_typicus 0.10×, Mus_musculus 0.27×, Ceratotherium_simum 0.36×. So the v1 per-species lowercase variance (1.9–20%) reported in iter-2 is *mostly* this recipe × species-conservation-profile bias, with a smaller real GCF-vs-GCA assembly-source component on top.

## Fix options (proposed, not implemented)

### For N-stretches

| ID | option | cost | covers |
|---|---|---|---|
| **F1** *(recommended)* | Add `n_frac` (+ `longest_n_run`) columns and a `max_n_frac` config knob to [`dataset.smk`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/dataset.smk); drop rows above threshold before sharding. | minimal — one rule edit | M1+M2+M3+M4, mechanism-agnostic |
| F2 | Pre-extraction species-side N-filter (build `undefined.bed` from each `{species}.2bit`, `bedtools intersect -v` per-species projection BED). Mirrors the human-side filter. | moderate pipeline change | M1+M2+M3 |
| F3 | Tighten resize policy (cap expansion length / require minimum lift length). | small | M2 only |
| F4 | Contig-size guard. | small | M3 only |

**Recommendation: F1.** At ~0.5% row loss it's near-free, and Section B step 4 showed mechanisms are mixed enough that a single mechanism-targeted fix wouldn't be definitively better.

### For repeat-masking

| ID | option | cost | comments |
|---|---|---|---|
| **R1** *(recommended)* | Document per-assembly variation in the dataset card; accept as-is. v1's mask is already what a uniform-RepeatMasker pipeline produces given current Dfam coverage (iter-5). | nil | |
| R5 | Re-mask each species with newer **Dfam 3.9** (Zoonomia used 3.3 from 2018). Catches a few more pp of repeats in under-represented lineages. | ~3 h wall on c6id.4xlarge; worth a 1-species spike first | might trim the GCF/GCA gap by a few pp |
| ~~R2~~ | ~~Uniform RepeatMasker pass.~~ **Superseded** — already done by the Zoonomia protocol (Dfam 3.3 + cactus-preprocess RED). | n/a | |
| R3 | Per-species RepeatMasker BEDs from UCSC/Ensembl/NCBI. | medium | heterogeneous track availability |
| ~~R4~~ | ~~Re-extract from NCBI Datasets FASTAs.~~ **Rejected** in [iter-4](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434656379) — would *decrease* mask coverage for 3 of 4 species tested. | n/a | |

**Recommendation: R1.** R5 is the only non-trivial option still on the table, and only worth it if downstream evals surface a real GCF/GCA training asymmetry.

## Settled / deferred

- Does the training pipeline distinguish case? **Yes — `a` and `A` tokenize identically but repeats are downweighted 100×.** This is what made the masking-uniformity question meaningful at all. The per-species effective weight (0.78–0.98) and 10 pp GCF/GCA gap are the practical numbers; both small enough to accept (iter-5).
- Cactus masking provenance: **uniform RepeatMasker pre-mask (Dfam 3.3, target 40-50% per assembly) followed by `cactus-preprocess` RED** (alignment-based, adds on top). Sourced from the 241-mammalian-2020v2.1 README and the cactus progressive docs. The iter-4 "HAL > NCBI Datasets" pattern is RED contributing the surplus on top of uniform Zoonomia-protocol RepeatMasker.
- 108-species genome-wide HAL lowercase pass — not done; deferred. The 4-species result was consistent enough with the protocol-documented 40-50% target that a full pass wasn't load-bearing for the recommendation.
- F1 threshold — to be picked when F1 is implemented (separate PR). 5% / 10% / 50% all defensible; whatever's chosen, log the dropped-row count.
- R5 Dfam 3.3 → 3.9 spike — deferred. Only worth it if downstream evals surface a real GCF/GCA training asymmetry.

## Tracking

Closed; reopen if downstream eval data reveals the masking variance actually moves model behaviour. Comments = append-only iteration log (`🤖` prefix, commit-pinned permalinks).

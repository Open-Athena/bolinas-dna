## Question

Two anomalies were observed in the public [`bolinas-dna/zoonomia-v1-v1`](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-v1) HuggingFace training dataset that warrant investigation:

1. **N-stretches in training windows.** Example row (page 2 of v1):

   ```
   win_7_000231918  Ceratotherium_simum  JH767724.1  24,562,726  24,562,981  +  78,580,661
   NNNN…NNN (127 leading N's) CGCCGCCGCCGCG…
   ```

   A 255 bp window with ~127 contiguous N's at its 5′ end. The first thing to rule out is **a bug in our region-expansion code** (`bolinas.projection.resize.resize_to_length` adding flanking bases beyond what `halLiftover` returned). Other plausible non-bug causes: the species genome has a real scaffold-internal assembly gap, the lifted region landed at a contig boundary, the target contig itself is tiny, or something else we haven't anticipated.

2. **Repeat-masking less consistent than RefSeq.** The prior RefSeq-based [`training_dataset/dataset_creation`](https://github.com/Open-Athena/bolinas-dna/tree/a38154132b173bffe0fb5d589682656b327759fd/snakemake/training_dataset/dataset_creation) pipeline produced sequences with reliable soft-masking (lowercase = RepeatMasker-flagged). v1 looks much less uniformly masked. The user's read (worth verifying) is that this is **upstream-driven** — the Zoonomia 447 Cactus alignment ingested per-leaf assemblies with varying RepeatMasker coverage, while NCBI Datasets applies a uniform masking pipeline per accession. Per-species amount and consistency should both be quantified.

## Scope

**In:**

- Trace the example anchor end-to-end against the rhino 2bit (`results/projection/genomes/Ceratotherium_simum.2bit` on S3) — re-extract, inspect surrounding gap context, distinguish "lift in gap" vs "resize expansion into gap" vs "tiny contig" mechanisms.
- Stream the source [`all_species_with_sequence.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/project.smk#L316-L335) and compute, per species:
  - N stats: mean N-fraction, fraction-with-any-N, fraction-with-N-frac≥{0.1, 0.5}, mean longest N-run, N-position-within-window decomposition (5′-edge / 3′-edge / interior).
  - Lowercase fraction stats: mean, p10/p50/p90 across windows.
- Join both with the species TSV ([`config/species_zoonomia_447_family_dedup.tsv`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/config/species_zoonomia_447_family_dedup.tsv)) on `assembly_level`, `contig_n50`, `quality_source`, and the `GCF_` vs `GCA_` accession prefix.
- Baseline lowercase fraction from the prior RefSeq HF dataset (sampling the matched genome set).
- A uniform RepeatMasker pass on a representative subsample of v1 sequences (e.g. 1–10 k windows × ~6 species spanning the GCF/GCA × quality-source grid). Compare RM-flagged bases to v1's existing lowercase: concordance + gap.
- Verify `hal2fasta` flag/source behavior regarding mask preservation through HAL→FASTA.
- **Fix options written into this issue body**, no implementation in this thread.

**Out (deferred):**

- Implementing any of the fix options. Each likely warrants its own PR or follow-up issue.
- Re-running `halLiftover` on the full HAL to recover per-window pre-resize spans. Only triggered if dataset-wide stats can't distinguish M1 vs M2 dominance.
- Rebuilding the v1 dataset. The current dataset stays as-is until a fix is decided.

## Approach

**Compute environment** — single SkyPilot c6id.4xlarge in `us-east-2`, tagged `project=dna` (per `CLAUDE.md` cloud conventions and `cloud_launch_conventions` memo). One `sky launch` at session start, reused via `sky exec` for each iteration; `sky down` only at session end.

**Branch / SHA** — [`claude/stoic-sutherland-ba71dd`](https://github.com/Open-Athena/bolinas-dna/tree/claude/stoic-sutherland-ba71dd); starting at [`a381541`](https://github.com/Open-Athena/bolinas-dna/commit/a38154132b173bffe0fb5d589682656b327759fd) (= `main`). All scratch analysis scripts live under a gitignored `scratch_audit/` directory at the worktree root (per `feedback_temp_files` memory).

**Key code paths under investigation** (all SHA-pinned to `a381541`):

- N-filter for human-side anchors: [`windows.smk`#L10-L34](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/windows.smk#L10-L34) + [`genome.smk`#L56-L66](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/genome.smk#L56-L66) (no analog for species side).
- Resize (potential expansion source): [`src/bolinas/projection/resize.py`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/src/bolinas/projection/resize.py).
- Post-projection filters (no sequence inspection): [`src/bolinas/projection/filter.py`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/src/bolinas/projection/filter.py).
- `hal2fasta` + `bedtools getfasta -s` extraction (no mask flags): [`project.smk`#L237-L313](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/project.smk#L237-L313).
- RefSeq reference: [`download.smk`#L1-L19](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/training_dataset/dataset_creation/workflow/rules/download.smk#L1-L19) (NCBI Datasets → `faToTwoBit`) and [`dataset.smk`#L22-L35](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/training_dataset/dataset_creation/workflow/rules/dataset.smk#L22-L35) (`twoBitToFa -bedPos`, default mask-preserving).

**Mechanisms-under-test for the N-stretches** (we want to *distinguish* them, not commit to one):

- **(M1) Lift inside a real assembly gap.** `halLiftover` returns a span landing in or across a scaffold-internal N-run.
- **(M2) Resize expansion into an adjacent gap.** Short lift gets midpoint-expanded outward, hitting a contig boundary.
- **(M3) Contig-edge effects.** Target contig is tiny → window covers contig + N-padding from scaffold context.
- **(M4) Something we haven't anticipated.** Kept open.

5′-edge / 3′-edge / interior N-position decomposition over the dataset should rank these.

## Current findings

_Pending: investigation cluster launching now; first iteration will land below._

## Open questions / next steps

- Does the example anchor's surrounding gap context in `Ceratotherium_simum.2bit` reveal an M2 (resize expansion into gap) or M1 (lift in gap) signature?
- What is the per-species N-fraction distribution across all 108 v1 species? Which assemblies are worst, and how does that correlate with `quality_source` / accession prefix?
- What is the per-species lowercase fraction distribution? Is it 0 across the board, or does it spread by `quality_source` / accession prefix as hypothesized?
- How does a uniform RepeatMasker pass compare to v1's existing lowercase in concordance and total mask coverage?
- Which fix options (F1–F4 for N's, R1–R4 for masking — listed once findings settle) should we plan to implement?

## Tracking

Description = current state. Comments = append-only iteration log (`🤖` prefix, commit-pinned permalinks). Fix proposals will be added to this body once findings allow ranking; no implementation in this thread.

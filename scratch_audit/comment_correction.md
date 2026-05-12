🤖 **Scope correction — RefSeq baseline dropped**

I mislabeled `genomes-v5-genome_set-mammals-intervals-v1_255_128` as a generic RefSeq baseline in [iter-2](https://github.com/Open-Athena/bolinas-dna/issues/177#issuecomment-4434090865). In the prior pipeline, **recipe `v1` = TSS-centered** (256 bp upstream + 256 bp downstream of TSS — promoters / CpG islands), per [`training_dataset/dataset_creation/README.md`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/training_dataset/dataset_creation/README.md). Promoter regions are systematically repeat-poor, so 14.1% lowercase there understates what NCBI's RepeatMasker produces genome-wide (typically ~40–50% for mammals).

More importantly: **no recipe in the RefSeq pipeline is composition-equivalent to zoonomia-v1-v1.** v1 zoonomia is conservation-filtered cross-mammal anchors, not a functional-region recipe. There's no fair single-number absolute-level comparison to publish.

**The headline finding doesn't change.** It was never the absolute level — it was the **per-assembly consistency**, which lives entirely inside v1:

- GCA leaves (73% of v1): mean lowercase 6.6%
- GCF leaves (27% of v1): mean lowercase 16.7%
- Per-species range: 1.9% → 20.0% (~10× spread)
- v1 `Homo_sapiens`: 1.9% vs Ensembl `dna_sm` ~50% — HAL → FASTA path strips most of the mask info even when present upstream

That bimodal-by-source pattern, plus the ~25× drop on human vs the upstream FASTA, is the "less consistent than RefSeq" signal directly. In the old pipeline, every leaf flows through the same NCBI Datasets → `genomic.fna` → `faToTwoBit` → `twoBitToFa` chain, so per-genome masking is governed by what NCBI applied per assembly — uniform within that pipeline by construction. In v1 the Cactus HAL acts as a many-input merge across heterogeneous assembly-submission practices.

Updating the issue body to drop the misleading "comparable" framing. Fix-option ranking (F1 / R4) is unaffected.

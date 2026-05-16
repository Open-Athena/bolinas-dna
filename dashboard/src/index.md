---
title: Bolinas-DNA Leaderboard
toc: false
---

# Bolinas-DNA Leaderboard

Public, version-controlled leaderboards for genomic language models trained under [Bolinas](https://github.com/Open-Athena/bolinas-dna), evaluated on matched-pair variant-effect prediction.

## Leaderboards

- [**Mendelian traits**](./leaderboards/mendelian) — OMIM ∪ HGMD ∪ Smedley pathogenic SNVs vs gnomAD common-frequency, gene-matched 1:1. Sort axis: Macro Avg.
- [**Complex traits**](./leaderboards/complex) — UKBB fine-mapped variants (`max(PIP) > 0.9`) vs non-fine-mapped, gene + MAF-matched. Sort axis: Global.
- [**eQTL**](./leaderboards/eqtl) — GTEx v8 fine-mapped eQTLs (49 tissues pooled, `max(PIP) > 0.9`) vs non-fine-mapped. Sort axis: Global.

## Protocols / Scoring approaches

A model family's PairwiseAccuracy depends on which score column you compute it from. These pages compare protocols head-to-head — same models, same dataset, different scoring approach.

- [**Bolinas**](./protocols/bolinas) — LLR vs JSD
- [**GPN-Star**](./protocols/gpn-star) — calibrated (cLLR) vs uncalibrated LLR

## Reference

- [**Models**](./models) — every entry on the leaderboards, with family / training / source links.
- [**About**](./about) — methodology, agent-readable data tier, "how to add a model" workflow.

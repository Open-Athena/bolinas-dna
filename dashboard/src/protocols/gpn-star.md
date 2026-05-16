---
title: GPN-Star — cLLR vs LLR
toc: false
---

# GPN-Star — calibrated vs uncalibrated LLR

*This page is a stub. The delta view (cells showing `+1.0` / `-2.5` for the change from cLLR to LLR) lands in the next pass.*

GPN-Star ships two variants of its per-variant score: the raw log-likelihood ratio (LLR) and a pentanucleotide-context-calibrated version (cLLR) — `llr_calibrated = llr − E[llr | 5-mer, mut]`. The [Mendelian leaderboard](../leaderboards/mendelian) defaults to cLLR (the producer's recommendation); the uncalibrated LLR is what most external papers cite.

This page visualises *which of the three V/M/P variants is most affected by the calibration, on which subsets*.

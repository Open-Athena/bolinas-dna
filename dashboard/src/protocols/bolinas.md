---
title: Bolinas — LLR vs JSD
toc: false
---

# Bolinas — LLR vs JSD

*This page is a stub. The delta view (cells showing `+1.0` / `-2.5` for the change from LLR to JSD) lands in the next pass.*

Bolinas gLMs assign a log-likelihood ratio (LLR) to each `(ref, alt)` variant — the standard scoring approach used on the [Mendelian leaderboard](../leaderboards/mendelian). An alternative is Jensen-Shannon divergence of the next-token distribution (JSD), which captures local context shifts without committing to a directionality.

When the bolinas/JSD toggle is flipped on the leaderboard, the rows reshuffle. This page visualises *which models moved, by how much, on which subsets*.

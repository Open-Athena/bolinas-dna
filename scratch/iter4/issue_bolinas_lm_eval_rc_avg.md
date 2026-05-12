## Context

#175 (iter-4 RC-strand investigation) showed that AVG of FWD + RC strand scores beats single-strand on most (model, subset) cells in matched-pair PairwiseAccuracy. Filed in biofoundation for the offline batched path: Open-Athena/biofoundation#24. This issue tracks the parallel change for the **online lm_eval harness** that runs during training (e.g. exp166's TraitGym Mendelian eval every 2733 steps) and feeds the matched-pair leaderboards #161 / #162 / #172.

The online lm_eval path is likelihood-only — embedding-distance scores aren't supported, so this issue is scoped to LLR-family scores.

## Requirement

The online lm_eval VEP scoring path used during training should report per-variant LLR-family scores that are the average of FWD-strand and RC-strand evaluations on the same variant.

## Rationale

See #175 for the full data. Headline: exp58-mammals × splicing × `minus_llr` — FWD=0.782, RC=0.731, **AVG=0.846** (+0.064 over best single strand). Reproduced in exp58-animals (+0.051). AVG-by-default during training gives the eval loop a tighter signal at every checkpoint, particularly on splicing variants where the CDS-vs-intron boundary effect makes FWD and RC give near-orthogonal evidence.

## Leaderboard implications

The matched-pair leaderboards (#161, #162, #172) currently report `minus_llr` (mendelian) and `abs_llr` (complex_traits, eqtl) on FWD-strand scoring. **Switching to AVG would change all historical leaderboard numbers by a few percent.** Two options:

1. **Hard switch**: rerun all leaderboard entries with AVG; document the change in the leaderboard headers.
2. **Add as additional column**: report both FWD and AVG accuracy per entry.

Discuss before changing.

## Acceptance criteria

- During a training run with this enabled, the lm_eval eval reports AVG PairwiseAccuracy on the LLR-family scores it currently computes.
- The reported AVG values match offline runs of the same (model, dataset, window).

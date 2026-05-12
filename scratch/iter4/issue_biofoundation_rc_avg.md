## Context

Open-Athena/bolinas-dna#175 (iter-4 RC-strand investigation, 5 published models × mendelian × win=256/255) showed that averaging FWD and RC strand scores is a strict-or-tied improvement over single-strand scoring on the matched-pair PairwiseAccuracy metric. The current offline VEP path (`compute_llr_and_embedding_distance` / `run_llr_and_embedding_distance`) scores only the FWD strand, leaving this on the table.

## Requirement

The VEP scoring path should be able to produce per-variant scores that are the average of FWD-strand and RC-strand evaluations on the same variant. The RC version must use the reverse-complement of the centered window, with the variant placed at the RC-space DNA index, and the allele index complemented. Averaging is applied after the sign convention so the average is over "signed" scores.

## Rationale (data from iter-4)

Headline same-score same-subset results (`minus_llr`):

| model | subset | FWD | RC | AVG | Δ(AVG − best strand) |
|---|---|---:|---:|---:|---:|
| exp58-mammals | splicing | 0.782 | 0.731 | **0.846** | **+0.064** |
| exp58-animals | splicing | 0.744 | 0.731 | **0.795** | +0.051 |
| exp58-animals | synonymous | 0.636 | 0.606 | **0.697** | +0.061 |
| exp58-mammals | missense | 0.775 | 0.776 | **0.795** | +0.019 |

Across 5 models × 8 mendelian subsets:
- AVG wins or ties on most home-domain (model, subset) cells.
- Where one strand strongly dominates (e.g. exp55-mammals 5'UTR has FWD=0.828 vs RC=0.713), AVG underperforms FWD by ≤0.02 — small downside.
- The CDS-vs-non-CDS boundary effect (most pronounced on splicing) means FWD and RC give fundamentally different evidence; AVG aggregates them.

Full data: Open-Athena/bolinas-dna#175 thread, especially comments [`-4434022140`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434022140) (4-model bug-check) and [`-4434107699`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434107699) (CDS-boundary hypothesis confirmation).

## Constraint

AVG only makes sense for score families where FWD and RC are comparable: LLR-family scores and embedding distances over `*_flat_*` / `*_mean_*` / `*_lastpos_last` pools.

**AVG must be skipped for `*_varpos_*` and `*_lastpos_middle` pools** — those pick a single token whose left-context window is entirely different between FWD and RC, so AVG over them is mathematically meaningless. The `embed_cosine_varpos_last` score is even anti-correlated between strands (pooled Pearson −0.26).

## Acceptance criteria

- FWD-only behavior is preserved (no regression vs current default).
- On exp58-mammals × mendelian × `minus_llr`, FWD-only matches iter-1 numbers and AVG matches iter-4: missense FWD=0.775 / AVG=0.795; splicing FWD=0.782 / AVG=0.846.
- For non-AVG-compatible score families, AVG is either silently rejected or returns the FWD score with a warning — explicit decision documented either way.

## Cost

Averaging doubles compute per variant (one extra forward pass). No memory increase. Acceptable for offline batched evals.

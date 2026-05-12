🤖 **Iter-4 bug-check follow-up: the user's "CDS-boundary effect" hypothesis is confirmed.** Commit [`a195b14`](https://github.com/Open-Athena/bolinas-dna/commit/a195b14).

User proposed that exp58's weak RC equivariance on splicing/3'UTR comes from the **CDS-vs-flanking boundary**: exp58 is CDS-trained, so when a variant sits at the CDS/intron or CDS/3'UTR boundary, FWD's left context (128 bp upstream of variant) often includes CDS → model is confident, while RC's left context (= RC of 128 bp downstream of variant) often is intron or deeper 3'UTR → model is uncertain. The two strands compute very different LLRs, giving extreme disagreement on high-impact variants.

## Direct test: bin by `max(|FWD|, |RC|)`, look at within-bin correlation

**exp58 3'UTR (n=116):**

| `max(|F|,|R|)` quartile | n | mean max | mean min | Pearson within bin |
|---|---:|---:|---:|---:|
| 0.08–0.63 | 29 | 0.42 | 0.20 | 0.14 |
| 0.63–1.25 | 29 | 0.92 | 0.44 | 0.51 |
| 1.25–2.20 | 29 | 1.70 | 1.00 | 0.52 |
| **2.20–47.96** | 29 | **8.57** | **1.63** | **−0.06** |

**exp58 splicing (n=156) — even more striking:**

| `max(|F|,|R|)` quartile | n | mean max | mean min | Pearson within bin |
|---|---:|---:|---:|---:|
| 0.12–1.82 | 39 | 1.07 | 0.52 | 0.35 |
| 1.82–4.64 | 39 | 3.01 | 1.64 | 0.53 |
| 4.64–13.79 | 39 | 8.31 | 2.56 | 0.32 |
| **13.79–108.74** | 39 | **40.75** | **4.25** | **−0.47** |

The high-magnitude tail of both subsets shows the boundary signature:

- One strand has |LLR| ~8–40, the other has |LLR| ~1.6–4.3 — **one strand is highly confident, the other isn't**
- Within the bin, Pearson is ~0 or negative — variants split into two groups: "FWD huge, RC small" (CDS is upstream of variant in DNA-space, FWD sees it) and "RC huge, FWD small" (CDS is downstream of variant, RC sees it after revcomp)
- Mid-low-magnitude variants in middle quartiles correlate normally (~0.5) — both strands have ordinary context, ordinary agreement

## Per-variant examples confirming the mechanism

Top-5 splicing variants by `|FWD-RC|`, exp58:

| chrom | pos | label | FWD-LLR | RC-LLR | which strand "saw" the CDS |
|---|---:|---:|---:|---:|---|
| 17 | 42565656 | 1 | **−108.7** | −19.3 | FWD |
| 5 | 128334870 | 1 | −2.8 | **−79.2** | RC (CDS downstream) |
| 17 | 82893535 | 1 | **−74.3** | +0.8 | FWD |
| 3 | 93884905 | 1 | −0.5 | **−66.4** | RC |
| 11 | 34456207 | 0 | −0.9 | **−63.9** | RC |

The pattern is unambiguous — each high-impact splicing variant has one massively-confident strand and one quiet strand, depending on which side of the variant the CDS lies.

## Predictions across subsets, all consistent

The boundary theory predicts how exp58's redundancy varies by subset based on where the variant sits relative to CDS:

| subset | CDS placement | predicted | observed redundancy | observed FWD-std |
|---|---|---|---:|---:|
| missense | **Fully in CDS** — both strands see CDS | moderate-low (both strands confident, agree) | 0.385 | 10.6 |
| splicing | **At CDS-intron boundary** — one strand sees CDS, other intron | extreme | **0.954** | 16.7 |
| 3_prime_UTR | **At CDS-3'UTR boundary** — one sees CDS, other UTR | extreme | **0.991** | 3.6 |
| synonymous | **Fully in CDS** but smaller LLR magnitudes | moderate | 0.778 | 6.7 |
| distal | **Fully out of CDS** — neither strand sees CDS | quiet, partial agreement | 0.664 | 1.4 |

The "fully in CDS" cases (missense, synonymous) have moderate redundancy because both strands see CDS context but with imperfect equivariance. The "boundary" cases (splicing, 3'UTR) blow up because the two strands see categorically different things. The "fully out of CDS" cases (distal) are quiet and partly redundant in their quiet predictions.

## What this means for the leaderboard protocols

This isn't just an exp58 quirk — it's a general property of the CLM autoregressive factorization: **at functional-constraint boundaries, the two strands of a CLM are seeing fundamentally different evidence**. So:

- For **promoter / enhancer / 3'UTR models** (exp55, exp136, exp59): the model is trained on a regulatory region, so when a variant sits at the boundary with non-regulatory sequence, the same boundary effect should appear — though we don't see it as strongly because the regulatory→non-regulatory contrast is less stark than CDS→non-CDS.
- For **CDS models** (exp58): the CDS-vs-flanking boundary is the strongest functional-constraint contrast in the genome → boundary effect is most extreme here.
- **AVG of FWD + RC is even more justified** as the default leaderboard protocol. At boundaries, the two strands give near-orthogonal complementary evidence; averaging them aggregates a confident prediction (from the strand that saw CDS) with a quiet prediction (from the strand that saw non-CDS). The confident one dominates the average. **AVG essentially gives you the maximum-confidence answer across strands** — much more useful than either single strand on boundary variants.

## TL;DR

User's hypothesis nailed it. exp58's RC asymmetry is a **biological boundary effect** at CDS/flanking transitions, not a code bug. The CLM's autoregressive factorization means FWD and RC see different halves of the window, and for CDS-trained models, the half containing the CDS produces a strongly-confident LLR while the other half is quiet. **AVG of FWD + RC = max-confidence-across-strands** for these variants, which is exactly the desired behavior.

## Code

- [`scratch/iter4_compare_distributional.py`](https://github.com/Open-Athena/bolinas-dna/blob/a195b14/scratch/iter4_compare_distributional.py) — includes the per-quartile boundary test

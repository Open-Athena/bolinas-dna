🤖 **Iter-4 follow-up: exp58-animals comparison.** Ran RC scout on the other CDS-trained model (exp58-animals, vs exp58-mammals) on mendelian × win=256 to test whether the splicing-redundancy and 3'UTR-redundancy patterns are general to "CDS models" or specific to exp58-mammals. Commit [`658d521`](https://github.com/Open-Athena/bolinas-dna/commit/658d521).

## TL;DR

1. **Splicing redundancy is the same in both CDS models** (0.962 animals vs 0.954 mammals) and shows the same top-quartile anti-correlation signature (Pearson −0.54 in the high-magnitude bin). The CDS-vs-intron boundary effect is **robustly reproduced**.
2. **On all other subsets, exp58-animals behaves much more like exp55/exp59/exp136 — i.e. normal moderate redundancy 0.25–0.77.** Especially 3'UTR drops from 0.991 (mammals) to 0.572 (animals), distal from 0.664 to 0.257.
3. **AVG wins more decisively on exp58-animals** than on any other model. On all three of its home-domain subsets (missense / synonymous / splicing), AVG is the best mode, with the biggest wins on splicing (+0.051) and synonymous (+0.061).
4. **exp58-mammals is uniquely anomalous beyond the splicing boundary effect.** Worth a closer look at its training recipe vs exp58-animals.

## Redundancy per subset, side-by-side

| subset | exp58-mammals | exp58-animals | Δ |
|---|---:|---:|---:|
| splicing | **0.954** | **0.962** | +0.008 |
| 5_prime_UTR_variant | 0.830 | 0.767 | −0.063 |
| synonymous_variant | 0.778 | 0.767 | −0.011 |
| 3_prime_UTR_variant | **0.991** | 0.572 | **−0.419** |
| distal | 0.664 | 0.257 | **−0.407** |
| non_coding_transcript_exon | 0.599 | 0.354 | −0.245 |
| tss_proximal | 0.446 | 0.273 | −0.173 |
| missense_variant | 0.385 | 0.304 | −0.081 |

Both CDS models converge on splicing redundancy ~0.96 — i.e. the FWD and RC strands are giving **essentially independent measurements** of splicing variants regardless of which CDS-training species set. This is the boundary effect's robust signature.

But the rest of the table shows exp58-animals normalizing onto the same redundancy range (0.25–0.77) as the other models. **exp58-mammals' 3'UTR=0.991, distal=0.664, ncRNA=0.599 are not generic CDS-model behavior** — they're specific to exp58-mammals.

## exp58-animals splicing boundary test (replicates the mechanism)

```
splicing (n=156), quartiles of max(|FWD|, |RC|):
  0.19–1.44:   mean_max=0.95,  mean_min=0.46,  Pearson in bin = +0.624
  1.44–2.73:   mean_max=1.98,  mean_min=1.02,  Pearson in bin = +0.616
  2.73–8.58:   mean_max=4.33,  mean_min=2.03,  Pearson in bin = +0.642
  8.58–70.71:  mean_max=27.9,  mean_min=3.66,  Pearson in bin = −0.536  ← top quartile flips negative
```

Same exact signature as exp58-mammals: high-magnitude tail splits into two groups (FWD-huge-RC-quiet vs RC-huge-FWD-quiet), depending on which side of the variant the CDS lies. **Mechanism reproduces in the sister model.**

## exp58-animals PairwiseAccuracy: AVG wins on home domains

| subset | n_pairs | FWD | RC | AVG | best | Δ(AVG − best strand) |
|---|---:|---:|---:|---:|---|---:|
| splicing | 78 | 0.744 | 0.731 | **0.795** | AVG | **+0.051** |
| synonymous | 33 | 0.636 | 0.606 | **0.697** | AVG | **+0.061** |
| missense | 4495 | 0.804 | 0.807 | **0.823** | AVG | +0.016 |
| 5'UTR | 87 | 0.586 | **0.598** | 0.586 | RC | −0.012 |
| tss_proximal | 61 | **0.656** | 0.574 | 0.639 | FWD | −0.016 |
| 3'UTR | 58 | 0.466 | **0.655** | 0.569 | RC | −0.086 |
| distal | 56 | 0.446 | 0.464 | 0.464 | AVG/RC tied | 0 |
| ncRNA | 42 | 0.571 | 0.548 | 0.571 | AVG/FWD tied | 0 |

AVG wins (or ties) on **5 of 8 subsets**, including all three home-domain subsets. The strongest wins are on splicing (+0.051) and synonymous (+0.061), both lining up with the boundary-effect prediction: where FWD and RC give independent informative evidence, averaging combines them.

## Refreshed 5-model scatter

![FWD vs RC LLR — 5 models × 8 subsets](https://raw.githubusercontent.com/Open-Athena/bolinas-dna/658d521/scratch/iter4/iter4_fwd_rc_scatter_4models.png)

Rows: exp55-mammals, exp58-mammals, exp58-animals, exp59-mammals, exp136-proj_v30. Columns are the 8 mendelian subsets.

- exp58-mammals (row 2): clouds on multiple subsets — anomalous everywhere.
- **exp58-animals (row 3): mostly diagonals, EXCEPT splicing (col 6)** which shows the same cloud as exp58-mammals splicing.
- Other rows: mostly clean diagonals.

The splicing cloud is the only consistent abnormality across CDS models. Everything else exp58-mammals does is its own weirdness.

## Reframed conclusions

- **The CDS-boundary effect on splicing is a robust property of CDS-trained CLMs**, reproduced cleanly in both exp58 variants. Splice variants are the canonical "one strand sees CDS, other sees intron" case → independent strand evidence → AVG benefits most.
- **exp58-mammals has additional anomalies beyond the splicing boundary effect** — its weak redundancy on 3'UTR, distal, ncRNA is not seen in exp58-animals. Worth investigating exp58-mammals' specific training recipe (data mixture, RC augmentation rate, training-stage-specific quirks).
- **AVG-by-default in leaderboard protocols is even more strongly justified** by the exp58-animals data: 5 of 8 subsets ≥ best single strand, with the biggest gains on splicing/synonymous (+0.05/+0.06). Cost is one extra forward pass.

## Code @ [`658d521`](https://github.com/Open-Athena/bolinas-dna/commit/658d521)

- [`scratch/iter4/iter4_rc_exp58-animals__win256__mendelian_traits.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/658d521/scratch/iter4) — exp58-animals RC scores
- Analysis run via `scratch/iter4_compare_distributional.py` + ad-hoc per-quartile boundary tests

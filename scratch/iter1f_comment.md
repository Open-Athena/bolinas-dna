🤖 **iter-1f: pair-aware vs std logreg head-to-head at high-D (wide C grid). The pair-aware lift was C-tuning, not loss-shape.**

Targeted head-to-head on mendelian high-D recipes, `logspace(-10, 0, 11)` C grid, chrom-grouped 3-fold OOF:

| recipe | dim | std logreg (wide C) | pair-aware (wide C) | Δ (pa − std) |
|---|---:|---:|---:|---:|
| `abs_delta` | 1920 | **0.7234** ± 0.019 | 0.7196 ± 0.020 | −0.004 |
| `sym_concat` | 3840 | 0.7074 ± 0.020 | **0.7164** ± 0.020 | +0.009 |

Both within ~½ SE of each other. **At high-D with proper C, pair-aware and standard logreg are statistically indistinguishable.** The +0.193 lift I reported for pair-aware × `sym_concat` in iter-1d was almost entirely *wide-C tuning*, not the ranking loss itself — std logreg gets the same lift on the same recipe:

| recipe | iter-1 BFS (C=1.0) std logreg | iter-1f wide-C std logreg | wide-C lift |
|---|---:|---:|---:|
| `abs_delta` | 0.6193 | 0.7234 | **+0.104** |
| `sym_concat` | 0.6016 | 0.7074 | **+0.106** |

Same recipes, same data, same classifier — just a wider C grid that lets the inner CV pick the right ridge strength.

### Net pair-aware verdict

* **Pair-aware loss ≈ standard logreg** on every feature set tested in iter-1c/d/e/f — within statistical noise on the 5-scalar, 2-scalar, and high-D (D=1920, D=3840) recipes.
* The intuition that "pair-aware should help because it matches the eval metric" is *correct in principle* but the gain is too small to see at this sample size (~5K train pairs per outer fold). BCE with `class_weight='balanced'` and well-tuned ridge converges to essentially the same linear classifier.
* **Iter-1 BFS used C=1.0 across all 5 classifiers; that single hparam choice cost 0.05-0.10 macro PA per recipe on high-D mendelian.** Lesson: never put a single value in a BFS hparam grid for a regularization knob spanning many orders of magnitude — even 3 values across the log range catches >90% of the gain.

### Where the high-D ceiling sits

With wide C, every high-D mendelian recipe I've tested plateaus around **macro 0.71-0.72**:

```
sym_concat × logreg   0.7074
sym_concat × pa       0.7164
abs_delta × logreg    0.7234
abs_delta × pa        0.7196
```

The 2-scalar **`logreg(embed_last_l2, minus_llr)` wide C = 0.7750** remains the mendelian champion. So:

* High-D supervised on pooled embeddings still loses to a tiny linear head on 2 zero-shot scalars.
* High-D supervised matches *zero-shot* `embed_last_l2` (0.7408) on global PA (0.74) but doesn't exceed it on macro.

### Why I didn't finish all 3 high-D recipes

Killed iter-1f at 30 min after `sym_concat` + `abs_delta` because `mean_delta` (D=5760) on std logreg/liblinear was going to take another ~30 min and the answer for sym_concat / abs_delta already settles the pair-aware-vs-std question. Adding `mean_delta` wouldn't change the headline.

### Updated cluster cost

`supervised-vep-cpu` torn down before this run. iter-1f + the targeted abs_delta pair-aware run together: ~40 min local CPU. All remaining iter-1 frozen-embedding probes are now exhausted.

### Next

Iter-2 LoRA fine-tuning is the only remaining lever to break the complex_traits / eqtl `embed_last_l2` ceilings (0.6541 / 0.5360). The mendelian ceiling at 0.7750 is also probably loose given LoRA could improve `embed_last_l2` itself.

Cited commit: [`55cb82b`](https://github.com/Open-Athena/bolinas-dna/commit/55cb82bb1e7a1ea3c7d9d44d8e2c97cdde2eebf9). Partial output: [`scratch/iter1f_partial.txt`](https://github.com/Open-Athena/bolinas-dna/blob/55cb82bb1e7a1ea3c7d9d44d8e2c97cdde2eebf9/scratch/iter1f_partial.txt).

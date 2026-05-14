🤖 **iter-2a (LoRA, complex_traits): ties frozen baseline, no decisive lift.**

3-fold chrom-grouped OOF, `exp166-p1B` + LoRA on q/v_proj, rank=16, lr=1e-4, batch=2, margin=50, normalize=False, epochs=1, pair-aware ranking loss on flat-L2 distance.

| metric | LoRA OOF | frozen `embed_last_l2` | Δ vs `embed_last_l2` | Δ vs `abs_llr` (0.5355 global / 0.5927 macro) |
|---|---:|---:|---:|---:|
| global PA | 0.5798 ± 0.021 | **0.5824** ± 0.021 | −0.003 | **+0.044** |
| macro PA | 0.6523 ± 0.032 | **0.6541** ± 0.032 | −0.002 | **+0.060** |

LoRA produces predictions essentially indistinguishable from the frozen baseline. With the score = `||flat(last_ref) − flat(last_alt)||₂` formulation (LoRA-init = frozen baseline exactly), the training updates over 1 epoch don't move the score distribution enough to change the ranking.

### Iter-2a sweep summary (fold 0 of complex_traits, single-fold hparam tuning)

**Sweep #1** (margin × normalize): 4 configs. All gave the same +0.005 inner-val gain but ~zero outer-test gain.

**Sweep #2** (batch / lr / rank / target modules): 5 configs. Inner-val gains scaled with config (rank=16 best at +0.017, bs=8 at +0.011, qkvo HURT at −0.006). **But inner-val gains didn't generalize to outer-test** — outer-test global PA only moved 0.000 to +0.003 vs frozen, all within 1 SE.

Per-config table (1 epoch, fold 0):

| config | epoch-0 val_PA | epoch-1 val_PA | Δ inner-val | outer-test global PA | Δ vs frozen 0.5135 |
|---|---:|---:|---:|---:|---:|
| **rank=16** (winner inner-val) | 0.5659 | **0.5824** | +0.017 | 0.5169 | +0.003 |
| bs=8 | 0.5659 | 0.5769 | +0.011 | 0.5135 | 0.000 |
| bs=8 lr=3e-4 | 0.5659 | 0.5769 | +0.011 | 0.5135 | 0.000 |
| bs=2 accum=8 | 0.5659 | 0.5714 | +0.005 | 0.5169 | +0.003 |
| qkvo (Q/K/V/O target) | 0.5659 | 0.5604 | −0.006 | 0.5135 | 0.000 |

The fold-0 outer-test chroms (13, 19, 5, X) are inherently harder than the train-split average — frozen `flat_l2` on those chroms is 0.5135 (vs 0.5824 global). So the chrom-level variance dominates any LoRA signal at this sample size.

### Lessons / next

- **Sample size hypothesis**: complex_traits has 564 pairs (≈ 376 inner-train per fold). LoRA-rank=4 adds 584K trainable params; rank=16 adds 2.3M. At this n vs p, the chrom-grouped OOF is genuinely hard — model memorizes training-chrom-specific cues that don't transfer.
- **No critical loss-formulation issue**: the L2-flat distance ranking loss with the right margin (50 for unnormalized scale ~100) is well-behaved. Train loss curves are sensible.
- **Moving to eqtl now** (n=2306 pairs, 4× bigger). If LoRA can ever beat the frozen ceiling, it'll be where the train set is large enough to learn signal that generalizes across chroms.

Iter-2a complex_traits artifacts: [`scratch/iter2_lora_sweep.py`](https://github.com/Open-Athena/bolinas-dna/blob/16c64e5/scratch/iter2_lora_sweep.py), [`scratch/iter2_lora_sweep2.py`](https://github.com/Open-Athena/bolinas-dna/blob/16c64e5/scratch/iter2_lora_sweep2.py). OOF parquet on S3 (`s3://oa-bolinas/snakemake/analysis/supervised_vep/results/lora_oof/exp166-p1B/complex_traits.parquet`).

🤖 **iter-2 wrap-up: LoRA fine-tuning of `exp166-p1B` doesn't beat the frozen-embedding ceiling on complex_traits or eqtl.**

Iter-2 explored two LoRA architectures on top of `exp166-p1B`, both trained with pair-aware ranking loss and 3-fold chrom-grouped CV.

### Iter-2a — no-head LoRA, score = `||flat(last_ref) − flat(last_alt)||₂`

LoRA adapters on q/v_proj are the only learnable parameters. At LoRA init the score is identical to the frozen `embed_last_l2` baseline.

**complex_traits 3-fold OOF (rank=16, lr=1e-4, batch=2, margin=50, nonorm, epochs=1)**

| metric | LoRA OOF | frozen `embed_last_l2` | Δ |
|---|---:|---:|---:|
| global PA | 0.5798 ± 0.021 | **0.5824** ± 0.021 | −0.003 |
| macro PA | 0.6523 ± 0.032 | **0.6541** ± 0.032 | −0.002 |

Detailed sweep history: [`#180 comment 4450851925`](https://github.com/Open-Athena/bolinas-dna/issues/180#issuecomment-4450851925). 11 single-fold hparam configs spanning margin (0.1–50), batch (2–8), grad-accum (1, 8), lr (1e-4, 3e-4, 1e-3), rank (4, 8, 16), target modules (q/v vs q/k/v/o), normalize (true/false). Inner-val gains up to +0.017 but never generalized to outer-test (max outer-test lift was +0.003, within ±1 SE).

**eqtl fold-0 sweep (same 5 configs as complex_traits sweep #2)**

| config | outer-test global PA | Δ vs frozen `embed_last_l2` (0.5306) |
|---|---:|---:|
| bs8 | 0.5356 | +0.005 |
| bs2_accum8 | 0.5340 | +0.003 |
| bs8_lr3e4 | 0.5283 | −0.002 |
| rank16 | 0.5291 | −0.001 |
| qkvo | 0.5348 | +0.004 |

All within ±1 SE (±0.020) of frozen baseline. 3-fold OOF skipped — no signal to confirm. Same pattern as complex_traits: inner-val gains in [0, +0.004], outer-test gains all within noise.

### Iter-2b — LoRA + MLP head on `|mean(alt) − mean(ref)|`

Forward both contexts through LoRA-adapted backbone, mean-pool last hidden state, take the symmetric `|alt_pool − ref_pool|`, run through MLP `D → D/4 → 1`. LoRA-init no longer matches frozen baseline (head is random at init).

**complex_traits fold-0 sweep (6 head configs)**

| config | epoch-0 train_PA | epoch-1 train_PA | epoch-1 val_PA | outer-test global PA | Δ vs frozen 0.5135 |
|---|---:|---:|---:|---:|---:|
| head_m1 (margin=1.0) | 0.5046 | 0.6277 | 0.5385 | **0.5135** | 0.000 |
| head_m05 (margin=0.5) | 0.5046 | 0.6369 | 0.5385 | 0.5068 | −0.007 |
| head_lr3e4 | 0.5046 | 0.6431 | 0.5055 | 0.5068 | −0.007 |
| head_lr1e3 | 0.5046 | 0.6923 | 0.5165 | 0.5068 | −0.007 |
| head_rank4 (rank=4 vs 16) | 0.4708 | 0.6308 | 0.4835 | 0.5068 | −0.007 |
| head_drop03 (dropout=0.3) | 0.5046 | 0.6462 | 0.5604 | 0.5000 | −0.014 |

Every config shows the classic overfitting signature — train_PA jumps +0.13 to +0.19 in one epoch while val_PA drops or stays flat. Outer-test PA ties (head_m1) or falls slightly below the frozen baseline. The mean-pool input destroys per-position information that the frozen `flat_l2` score preserves, and the added MLP capacity (≈ 0.9-2.3 M extra params on top of LoRA) overfits the ~330 train pairs per fold.

3-fold OOF skipped on iter-2b — same reasoning as eqtl no-head.

### Mendelian intentionally skipped

Zero-shot already does well on mendelian — `rank-mean(embed_last_l2, minus_llr)` reaches 0.7682 macro (iter-1b), and `logreg(embed_last_l2, minus_llr)` wide-C reaches 0.7750 macro / 0.7462 global (iter-1d). The marginal value of LoRA there is low and orthogonal to the question of whether LoRA can break frozen ceilings on the harder datasets.

### Final scoreboard

No changes vs the iter-1d state — LoRA didn't unseat any of the existing champions. The current best per dataset stays:

| dataset | best recipe so far | global PA | macro PA |
|---|---|---:|---:|
| `mendelian_traits` | `logreg(embed_last_l2, minus_llr)` wide C | 0.7462 ± 0.006 | 0.7750 ± 0.018 |
| `complex_traits` | zero-shot `embed_last_l2` | **0.5824** ± 0.021 | **0.6541** ± 0.032 |
| `eqtl` | zero-shot `embed_last_l2` | **0.5306** ± 0.010 | **0.5360** ± 0.022 |

### Compute

`supervised-vep-lora` (A10G:1, us-east-2), ~16 hours total wall-clock across iter-2a + iter-2b + the 3-fold OOFs. About 12 hrs of that was active GPU work; the rest was setup + investigation idle. Tearing down now.

### Code

- LoRA model + loss: [`src/bolinas/supervised/lora.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/src/bolinas/supervised/lora.py)
- Per-fold training entry point: [`src/bolinas/supervised/lora_pipeline.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/src/bolinas/supervised/lora_pipeline.py)
- Snakemake rules: [`snakemake/analysis/supervised_vep/workflow/rules/lora.smk`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/snakemake/analysis/supervised_vep/workflow/rules/lora.smk)
- Sweep scripts: [`scratch/iter2_lora_sweep.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/scratch/iter2_lora_sweep.py) (sweep #1), [`iter2_lora_sweep2.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/scratch/iter2_lora_sweep2.py) (sweep #2), [`iter2_lora_sweep_eqtl.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/scratch/iter2_lora_sweep_eqtl.py) (eqtl), [`iter2b_lora_head_sweep.py`](https://github.com/Open-Athena/bolinas-dna/blob/b66b430/scratch/iter2b_lora_head_sweep.py) (head).
- OOF predictions on S3: `s3://oa-bolinas/snakemake/analysis/supervised_vep/results/lora_oof/exp166-p1B/complex_traits.parquet`.

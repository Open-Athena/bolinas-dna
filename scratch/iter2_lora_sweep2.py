"""Iter-2a sweep #2: vary batch size / lr / rank / target modules.

Sweep #1 found:
- All 4 (margin × normalize) configs produced the same +0.005 inner_val gain.
- nonorm + margin=50 had the highest absolute val_PA and slight +0.003 outer-test
  gain vs frozen flat_l2 baseline.
- Hypothesis: batch=2 is too noisy for a stable gradient signal. Try larger
  effective batches via direct batch increase (bf16+ckpt may have headroom) and
  via gradient accumulation.

Fix: nonorm + margin=50 + 1 epoch. Vary: effective batch, lr, rank, target modules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from bolinas.supervised.lora_pipeline import fit_predict_one_fold


GENOME_PATH = "/home/ubuntu/genome.fa.gz"  # stable cluster-local copy (not in snakemake storage cache)
HF_DATASET = "bolinas-dna/evals_complex_traits"
SPLIT = "train"
BACKBONE = "bolinas-dna/exp166-p1B-step-16398"
WINDOW = 255
FOLD = 0

# Anchor: nonorm + margin=50 + 1 epoch (best from sweep #1).
COMMON = dict(
    lora_rank=4, lr=1e-4, batch_size=2, grad_accum_steps=1, epochs=1,
    margin=50.0, normalize=False,
)

CONFIGS = [
    # 1) Effective batch via direct bs increase (bf16+ckpt should give headroom).
    ("bs8",         {**COMMON, "batch_size": 8}),
    # 2) Effective batch via gradient accumulation (memory-safe at any batch).
    ("bs2_accum8",  {**COMMON, "grad_accum_steps": 8}),
    # 3) Higher lr with larger effective batch.
    ("bs8_lr3e4",   {**COMMON, "batch_size": 8, "lr": 3e-4}),
    # 4) Higher capacity.
    ("rank16",      {**COMMON, "lora_rank": 16}),
    # 5) More target modules (Q/K/V/O instead of just Q/V).
    ("qkvo",        {**COMMON, "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj")}),
]


def main():
    out_dir = Path("/home/ubuntu/iter2_sweep2_out")
    out_dir.mkdir(exist_ok=True, parents=True)
    rows = []
    for label, hp in CONFIGS:
        print(f"\n========== sweep#2 config: {label} ==========")
        print(f"  hparams: {hp}")
        t0 = time.time()
        try:
            preds, stats = fit_predict_one_fold(
                hf_dataset_path=HF_DATASET,
                split=SPLIT,
                backbone_id=BACKBONE,
                window_size=WINDOW,
                genome_path=GENOME_PATH,
                fold=FOLD,
                **hp,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append(dict(label=label, hp=str(hp), status="error", error=str(e)))
            continue
        elapsed = time.time() - t0
        rows.append(dict(
            label=label,
            hp=str(hp),
            status="ok",
            epoch0_train_pa=stats["train_pa"][0],
            epoch0_val_pa=stats["val_pa"][0],
            epoch1_train_pa=stats["train_pa"][-1],
            epoch1_val_pa=stats["val_pa"][-1],
            train_loss=stats["train_loss"][-1],
            elapsed_s=elapsed,
        ))
        preds.to_parquet(out_dir / f"{label}_predictions.parquet", index=False)
        with open(out_dir / f"{label}_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(
            f"  elapsed={elapsed:.1f}s  "
            f"epoch0 val_PA={stats['val_pa'][0]:.4f}  "
            f"epoch1 val_PA={stats['val_pa'][-1]:.4f}  "
            f"epoch1 train_PA={stats['train_pa'][-1]:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "iter2_sweep2_summary.parquet", index=False)
    print("\n========== SWEEP#2 SUMMARY ==========")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

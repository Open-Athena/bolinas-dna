"""Iter-2a hparam sweep on complex_traits fold 0 (1 epoch each).

Single-epoch runs are fast (~2-3 min on A10G) and visibly reveal overfitting
direction. We sweep rank / lr / batch_size / margin keeping the rest fixed at
the conservative defaults from `config.yaml`. For each config: train one fold,
log epoch-0 (frozen) + epoch-1 val_PA. Best config(s) then run as full 3-fold
OOF in a subsequent step.

Usage on the cluster:
    cd snakemake/analysis/supervised_vep
    uv run --project ../../.. python ../../../scratch/iter2_lora_sweep.py
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
FOLD = 0  # always use fold 0 — sweep is for hparam comparison, not OOF.

# Pass 1: with `normalize=True` (score ∈ [0, 2]), sweep MARGIN over canonical
# metric-learning values {0.1, 0.5, 1.0}. Plus one unnormalized control with
# margin=50 (the only value of the unnormalized scale that's likely to give
# meaningful gradient on already-correct pairs).
CONFIGS = [
    # Normalized variants — canonical hparam range.
    ("norm_m01",   dict(lora_rank=4, lr=1e-4, batch_size=2, epochs=1, margin=0.1, normalize=True)),
    ("norm_m05",   dict(lora_rank=4, lr=1e-4, batch_size=2, epochs=1, margin=0.5, normalize=True)),
    ("norm_m1",    dict(lora_rank=4, lr=1e-4, batch_size=2, epochs=1, margin=1.0, normalize=True)),
    # Unnormalized control with margin matched to score scale (~100).
    ("nonorm_m50", dict(lora_rank=4, lr=1e-4, batch_size=2, epochs=1, margin=50.0, normalize=False)),
]


def main():
    out_dir = Path("/home/ubuntu/iter2_sweep_out")
    out_dir.mkdir(exist_ok=True, parents=True)
    rows = []
    for label, hp in CONFIGS:
        print(f"\n========== iter-2a sweep config: {label} ==========")
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
            print(f"  ERROR: {e}")
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
        # Save per-config artifacts.
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
    df.to_parquet(out_dir / "iter2_sweep_summary.parquet", index=False)
    print("\n========== SWEEP SUMMARY ==========")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

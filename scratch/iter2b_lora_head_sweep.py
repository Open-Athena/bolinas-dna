"""Iter-2b: LoRA + MLP head on |mean(alt) − mean(ref)|. Sweep on complex_traits fold 0.

Architecture:
- Forward both ref/alt contexts through LoRA-adapted backbone.
- Mean-pool last hidden state → [B, 2, D].
- feat = |alt_pool − ref_pool| → MLP head (D → D/hidden_div → 1) → scalar score.
- Symmetric in ref↔alt by construction.

Differences from iter-2a (no-head):
- LoRA-init no longer matches frozen baseline (head is random at init).
- The pool-then-diff destroys per-position info (vs flat-L2 which keeps it).
  Zero-shot pool_l2 baseline on complex_traits was 0.5903 macro / 0.5301 global
  (vs flat_l2 0.6541 / 0.5824) — there's room to recover if the head learns.

Frozen baselines on complex_traits (full train): embed_last_l2=0.5824 global / 0.6541 macro.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from bolinas.supervised.lora_pipeline import fit_predict_one_fold


GENOME_PATH = "/home/ubuntu/genome.fa.gz"
HF_DATASET = "bolinas-dna/evals_complex_traits"
SPLIT = "train"
BACKBONE = "bolinas-dna/exp166-p1B-step-16398"
WINDOW = 255
FOLD = 0

# Anchor: head=True, the best no-head config from iter-2a sweep#2 (rank=16),
# 1 epoch, batch=8 (sweep#2 winner for non-head). Margin doesn't apply to
# head output the same way — head output is unbounded but typically O(1).
# Try a few margin/rank/lr combos.
COMMON = dict(
    lora_rank=16, lr=1e-4, batch_size=8, grad_accum_steps=1, epochs=1,
    normalize=False, use_head=True, head_hidden_div=4, head_dropout=0.1,
)

CONFIGS = [
    ("head_m1",       {**COMMON, "margin": 1.0}),
    ("head_m05",      {**COMMON, "margin": 0.5}),
    ("head_lr3e4",    {**COMMON, "lr": 3e-4, "margin": 1.0}),
    ("head_lr1e3",    {**COMMON, "lr": 1e-3, "margin": 1.0}),
    ("head_rank4",    {**COMMON, "lora_rank": 4, "margin": 1.0}),
    ("head_drop03",   {**COMMON, "head_dropout": 0.3, "margin": 1.0}),
]


def main():
    out_dir = Path("/home/ubuntu/iter2b_head_out")
    out_dir.mkdir(exist_ok=True, parents=True)
    rows = []
    for label, hp in CONFIGS:
        print(f"\n========== iter-2b sweep config: {label} ==========")
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
    df.to_parquet(out_dir / "iter2b_head_sweep_summary.parquet", index=False)
    print("\n========== ITER-2B HEAD SWEEP SUMMARY ==========")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""CLI training script for the per-bin enhancer segmenter.

Usage::

    uv run python -m bolinas.enhancer_segmentation.train \\
        --train-parquet results/dataset/segmentation/seg_v1/train.parquet \\
        --val-parquet results/dataset/segmentation/seg_v1/validation.parquet \\
        --weights-path results/weights/model_all_folds.safetensors \\
        --output-ckpt results/model/segmentation/default/seg_v1/best.ckpt \\
        --output-metrics results/model/segmentation/default/seg_v1/metrics.json \\
        --output-val-predictions results/model/segmentation/default/seg_v1/val_predictions.parquet \\
        --wandb-run default-seg_v1
"""

import argparse
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
import wandb
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from bolinas.enhancer_segmentation.dataset import SegmentationDataset
from bolinas.enhancer_segmentation.model import EnhancerSegmenter

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")

WANDB_PROJECT = "bolinas-enhancer-segmentation"
WANDB_ENTITY = "gonzalobenegas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-bin enhancer segmenter")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--output-ckpt", type=str, required=True)
    parser.add_argument("--output-metrics", type=str, required=True)
    parser.add_argument("--output-val-predictions", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Lightning Trainer.max_steps (-1 = unlimited). "
        "Takes precedence over --limit-train-batches for the LR schedule.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction (<=1.0) of training batches per epoch, or an int number "
        "of batches. Mirrors Lightning Trainer.limit_train_batches.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction (<=1.0) or int number of validation batches. Mirrors "
        "Lightning Trainer.limit_val_batches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-run", type=str, required=True)
    return parser.parse_args()


def _as_batches_arg(v: float) -> int | float:
    """Interpret a CLI float as Lightning's int|float batches argument.

    Values >= 1 with no fractional part are treated as an integer count;
    values in (0, 1] stay as a fraction.
    """
    if v >= 1 and float(v).is_integer():
        return int(v)
    return float(v)


def compute_pos_weight(train_parquet: str) -> float:
    """Compute BCE pos_weight = (# negative bins) / (# positive bins) from the
    training parquet. Derives the value from the exact dataset the model sees
    so it cannot drift from the data (subsampling, splits, RC augmentation).
    """
    labels = pl.read_parquet(train_parquet, columns=["labels"])["labels"]
    arr = np.asarray(labels.to_list(), dtype=np.uint8)
    n_pos = int(arr.sum())
    n_neg = int(arr.size - n_pos)
    if n_pos == 0:
        raise ValueError("Training set has no positive bins; cannot set pos_weight")
    return n_neg / n_pos


def main() -> None:
    args = parse_args()
    output_ckpt = Path(args.output_ckpt)
    output_metrics = Path(args.output_metrics)
    output_dir = output_ckpt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    pos_weight = compute_pos_weight(args.train_parquet)
    log.info("Computed pos_weight=%.4f from training set", pos_weight)

    train_ds = SegmentationDataset(args.train_parquet, augment_rc=True)
    val_ds = SegmentationDataset(args.val_parquet, augment_rc=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # LR schedule must cosine over the steps actually taken, not the full
    # epoch — otherwise fast iteration runs decay way too slowly.
    full_steps = len(train_loader)
    if args.max_steps > 0:
        num_training_steps = args.max_steps
    elif 0 < args.limit_train_batches <= 1.0:
        num_training_steps = max(1, int(full_steps * args.limit_train_batches))
    elif args.limit_train_batches > 1.0:
        num_training_steps = min(full_steps, int(args.limit_train_batches))
    else:
        num_training_steps = full_steps

    model = EnhancerSegmenter(
        weights_path=args.weights_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        warmup_fraction=args.warmup_fraction,
        num_training_steps=num_training_steps,
        pos_weight=pos_weight,
    )

    model = torch.compile(model)

    loggers = [
        CSVLogger(output_dir, name="logs"),
        WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=args.wandb_run,
            save_dir=output_dir,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=args.max_steps,
        limit_train_batches=_as_batches_arg(args.limit_train_batches),
        limit_val_batches=_as_batches_arg(args.limit_val_batches),
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,  # single-GPU only; multi-GPU not supported
        gradient_clip_val=args.gradient_clip_val,
        logger=loggers,
        default_root_dir=output_dir,
    )

    wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
    wandb_run_id: str | None = None
    if wandb_logger is not None:
        wandb_run_id = wandb_logger.experiment.id

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(output_ckpt)

    metrics = {
        "val_auprc": float(trainer.callback_metrics.get("val_auprc", 0.0)),
        "pos_weight": pos_weight,
    }

    # Per-(window, bin) validation predictions — one row per bin so downstream
    # PR/offline analyses can compute per-species metrics and
    # recall-at-target-precision.
    logits_flat = model.val_logits.compute().cpu().numpy()
    val_meta = pl.read_parquet(
        args.val_parquet,
        columns=["genome", "chrom", "start", "end", "strand", "labels"],
    )
    num_bins = len(val_meta["labels"][0])
    # With limit_val_batches < 1.0 Lightning only iterates a prefix of the
    # (unshuffled) val loader, so logits cover the first K windows. Truncate
    # val_meta to match — the val parquet is already shuffled at build time
    # so the prefix is a random sample.
    n_windows_seen = len(logits_flat) // num_bins
    if n_windows_seen < len(val_meta):
        val_meta = val_meta.head(n_windows_seen)
    expected = len(val_meta) * num_bins
    assert len(logits_flat) == expected, (
        f"Logit count ({len(logits_flat)}) != "
        f"validation rows * num_bins ({len(val_meta)} * {num_bins} = {expected})"
    )

    bin_size = None
    if len(val_meta) > 0:
        window_size = int(val_meta["end"][0] - val_meta["start"][0])
        bin_size = window_size // num_bins

    labels_flat = np.asarray(val_meta["labels"].to_list(), dtype=np.uint8).reshape(-1)
    bin_idx = np.tile(np.arange(num_bins, dtype=np.int32), len(val_meta))
    window_starts = np.repeat(val_meta["start"].to_numpy(), num_bins)
    predictions = pl.DataFrame(
        {
            "genome": np.repeat(val_meta["genome"].to_numpy(), num_bins),
            "chrom": np.repeat(val_meta["chrom"].to_numpy(), num_bins),
            "start": window_starts,
            "end": np.repeat(val_meta["end"].to_numpy(), num_bins),
            "strand": np.repeat(val_meta["strand"].to_numpy(), num_bins),
            "bin_idx": bin_idx,
            "bin_start": window_starts + bin_idx * (bin_size or 0),
            "bin_end": window_starts + (bin_idx + 1) * (bin_size or 0),
            "label": labels_flat,
            "logit": logits_flat,
        }
    )

    if args.output_val_predictions:
        Path(args.output_val_predictions).parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(args.output_val_predictions)

    # Per-species AUPRC
    per_species: dict[str, float] = {}
    for genome in predictions["genome"].unique().sort().to_list():
        subset = predictions.filter(pl.col("genome") == genome)
        labels = subset["label"].to_numpy()
        probs = expit(subset["logit"].to_numpy())
        if len(np.unique(labels)) == 2:
            per_species[f"val_auprc/{genome}"] = float(
                average_precision_score(labels, probs)
            )
    metrics.update(per_species)

    if per_species and wandb_run_id is not None:
        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                id=wandb_run_id,
                resume="must",
            )
            run.log(per_species)
            run.log({"pos_weight": pos_weight})
            run.finish()
        except Exception:
            log.warning("Failed to log per-species metrics to W&B", exc_info=True)

    output_metrics.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

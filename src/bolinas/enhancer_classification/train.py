"""CLI training script for the enhancer classifier.

Usage::

    uv run python -m bolinas.enhancer_classification.train \
        --train-parquet data/train.parquet \
        --val-parquet data/validation.parquet \
        --weights-path alphagenome.pth \
        --output-ckpt results/model/debug/v3/best.ckpt \
        --output-metrics results/model/debug/v3/metrics.json
"""

import argparse
import json
import shutil
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from bolinas.enhancer_classification.dataset import EnhancerDataset
from bolinas.enhancer_classification.model import EnhancerClassifier

torch.set_float32_matmul_precision("medium")

WANDB_PROJECT = "bolinas-enhancer-classification"
WANDB_ENTITY = "gonzalobenegas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhancer classifier")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--output-ckpt", type=str, required=True)
    parser.add_argument("--output-metrics", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overfit-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-run", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_ckpt = Path(args.output_ckpt)
    output_metrics = Path(args.output_metrics)
    output_dir = output_ckpt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    train_ds = EnhancerDataset(args.train_parquet)
    val_ds = EnhancerDataset(args.val_parquet)

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

    model = EnhancerClassifier(
        weights_path=args.weights_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
    )

    model = torch.compile(model)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        filename="best",
        save_top_k=1,
    )

    callbacks = [checkpoint_cb]
    if args.overfit_batches == 0:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=5, mode="min")
        )

    loggers = [CSVLogger(output_dir, name="logs")]
    if args.wandb_run:
        loggers.append(
            WandbLogger(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=args.wandb_run,
                save_dir=output_dir,
            )
        )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        overfit_batches=args.overfit_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=output_dir,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Copy best checkpoint to the Snakemake-managed output path
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        shutil.copy2(best_ckpt, output_ckpt)

    # Write metrics summary
    metrics = {
        "val_auroc": float(checkpoint_cb.best_model_score or 0.0),
        "best_epoch": int(
            Path(best_ckpt).stem.split("epoch=")[-1].split("-")[0]
            if best_ckpt and "epoch=" in best_ckpt
            else trainer.current_epoch
        ),
    }
    output_metrics.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

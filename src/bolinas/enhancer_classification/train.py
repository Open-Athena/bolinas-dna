"""CLI training script for the enhancer classifier.

Usage::

    uv run python -m bolinas.enhancer_classification.train \
        --train-parquet data/train.parquet \
        --val-parquet data/validation.parquet \
        --weights-path alphagenome.pth \
        --output-dir results/model/linear_probe
"""

import argparse
import json
import shutil
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from bolinas.enhancer_classification.dataset import EnhancerDataset
from bolinas.enhancer_classification.model import EnhancerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhancer classifier")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
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

    if args.compile:
        model = torch.compile(model)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        filename="best",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=CSVLogger(output_dir, name="logs"),
        default_root_dir=output_dir,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Copy best checkpoint to a stable path
    best_ckpt = checkpoint_cb.best_model_path
    dest_ckpt = output_dir / "best.ckpt"
    if best_ckpt:
        shutil.copy2(best_ckpt, dest_ckpt)

    # Write metrics summary
    metrics = {
        "val_auroc": float(checkpoint_cb.best_model_score or 0.0),
        "best_epoch": int(
            Path(best_ckpt).stem.split("epoch=")[-1].split("-")[0]
            if best_ckpt and "epoch=" in best_ckpt
            else trainer.current_epoch
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

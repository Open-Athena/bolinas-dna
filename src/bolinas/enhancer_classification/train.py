"""CLI training script for the enhancer classifier.

Usage::

    uv run python -m bolinas.enhancer_classification.train \
        --train-parquet data/train.parquet \
        --val-parquet data/validation.parquet \
        --weights-path alphagenome.pth \
        --output-ckpt results/model/default/v3/best.ckpt \
        --output-metrics results/model/default/v3/metrics.json
"""

import argparse
import json
from pathlib import Path

import lightning as L
import polars as pl
import torch
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
    parser.add_argument("--output-val-predictions", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-run", type=str, required=True)
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

    num_training_steps = len(train_loader)

    model = EnhancerClassifier(
        weights_path=args.weights_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        warmup_fraction=args.warmup_fraction,
        num_training_steps=num_training_steps,
        mlp_hidden_dim=args.mlp_hidden_dim,
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
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,  # single-GPU only; multi-GPU not supported
        gradient_clip_val=args.gradient_clip_val,
        logger=loggers,
        default_root_dir=output_dir,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save checkpoint
    trainer.save_checkpoint(output_ckpt)

    # Write metrics summary
    metrics = {
        "val_auroc": float(trainer.callback_metrics.get("val_auroc", 0.0)),
        "val_auprc": float(trainer.callback_metrics.get("val_auprc", 0.0)),
    }
    output_metrics.write_text(json.dumps(metrics, indent=2))

    # Save per-sample validation predictions
    if args.output_val_predictions:
        output_val_predictions = Path(args.output_val_predictions)
        all_logits: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, _y = batch
                x = x.to(model.device)
                logits = model(x)
                all_logits.append(logits.cpu())
        logits_array = torch.cat(all_logits).numpy()

        val_meta = pl.read_parquet(
            args.val_parquet,
            columns=["genome", "chrom", "start", "end", "strand", "label"],
        )
        val_meta = val_meta.with_columns(pl.Series("logit", logits_array))
        val_meta.write_parquet(output_val_predictions)


if __name__ == "__main__":
    main()

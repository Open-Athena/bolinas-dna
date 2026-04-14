"""AlphaGenome per-bin enhancer segmenter."""

import logging
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from alphagenome_pytorch.model import SequenceEncoder
from transformers import get_cosine_schedule_with_warmup

from bolinas.enhancer_classification.model import (
    ENCODER_OUTPUT_DIM,
    load_pretrained_encoder,
)

log = logging.getLogger(__name__)


class EnhancerSegmenter(L.LightningModule):
    """Per-bin enhancer segmenter using AlphaGenome's CNN encoder trunk.

    Architecture: SequenceEncoder -> Conv1d(1536, 1, kernel_size=1) -> per-bin logits.
    The encoder downsamples 128x, so an input of shape (B, L, 4) yields
    (B, L/128) logits.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        freeze_backbone: bool = False,
        warmup_fraction: float = 0.1,
        num_training_steps: int | None = None,
        pos_weight: float = 1.0,
        genomes: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["weights_path"])
        self.genomes: list[str] = list(genomes) if genomes else []

        if weights_path is not None:
            self.encoder = load_pretrained_encoder(weights_path)
            log.info("Loaded pretrained AlphaGenome encoder from %s", weights_path)
        else:
            self.encoder = SequenceEncoder()
            log.info("No weights_path provided — encoder initialized from scratch")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # GroupNorm over channels (equivalent to LayerNorm on each position's
        # 1536-vector) before the final projection. Matches the classifier's
        # LayerNorm-before-Linear pattern and stabilizes gradient norms
        # (pre-norm an earlier run sat at mean=35 / max=90 with clip=1.0).
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=ENCODER_OUTPUT_DIM),
            nn.Conv1d(ENCODER_OUTPUT_DIM, 1, kernel_size=1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
        )
        # Overall AUPRC plus a per-species AUPRC metric per genome.
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc_per_species = nn.ModuleDict(
            {g: torchmetrics.AveragePrecision(task="binary") for g in self.genomes}
        )
        # Accumulate flat logits so train.py can write val_predictions.parquet.
        self.val_logits = torchmetrics.CatMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk, _intermediates = self.encoder(x)
        logits = self.head(trunk).squeeze(1)
        return logits

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, _g = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self.val_logits.reset()

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, y, g = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        flat_logits = logits.reshape(-1)
        flat_labels = y.reshape(-1).int()
        preds = torch.sigmoid(flat_logits)
        self.val_auprc.update(preds, flat_labels)
        self.val_logits.update(flat_logits)

        # Per-species AUPRC: each row of y belongs to genome g[row]; expand g
        # across bin positions to match the flattened logits/labels shape.
        num_bins = y.shape[1]
        g_flat = g.repeat_interleave(num_bins)
        for i, name in enumerate(self.genomes):
            mask = g_flat == i
            if mask.any():
                self.val_auprc_per_species[name].update(preds[mask], flat_labels[mask])

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)
        self.val_auprc.reset()
        for name, metric in self.val_auprc_per_species.items():
            # AveragePrecision raises if the metric never saw any sample; skip
            # genomes that don't appear in the val set.
            try:
                value = metric.compute()
            except Exception:
                metric.reset()
                continue
            self.log(f"val_auprc/{name}", value, prog_bar=False, sync_dist=True)
            metric.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        num_steps = self.hparams.num_training_steps
        if num_steps is None:
            raise ValueError("num_training_steps must be set for LR scheduling")
        num_warmup = int(self.hparams.warmup_fraction * num_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

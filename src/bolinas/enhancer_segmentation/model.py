"""AlphaGenome per-bin enhancer segmenter."""

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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["weights_path"])

        if weights_path is not None:
            self.encoder = load_pretrained_encoder(weights_path)
        else:
            self.encoder = SequenceEncoder()

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Conv1d(ENCODER_OUTPUT_DIM, 1, kernel_size=1)

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
        )
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_logits = torchmetrics.CatMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk, _intermediates = self.encoder(x)
        logits = self.head(trunk).squeeze(1)
        return logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
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
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        flat_logits = logits.reshape(-1)
        flat_labels = y.reshape(-1).int()
        preds = torch.sigmoid(flat_logits)
        self.val_auprc.update(preds, flat_labels)
        self.val_logits.update(flat_logits)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)
        self.val_auprc.reset()

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

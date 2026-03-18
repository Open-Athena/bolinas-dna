"""AlphaGenome convolutional-probe enhancer classifier."""

from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from alphagenome_pytorch.model import AlphaGenome, SequenceEncoder

ENCODER_OUTPUT_DIM = 1536


def load_pretrained_encoder(weights_path: str | Path) -> SequenceEncoder:
    """Load encoder weights from a full AlphaGenome checkpoint.

    Creates a full AlphaGenome model, extracts the encoder state_dict,
    loads it into a fresh SequenceEncoder, then discards the rest.
    """
    full_model = AlphaGenome.from_pretrained(weights_path, device="cpu")
    encoder_state = full_model.encoder.state_dict()
    del full_model

    encoder = SequenceEncoder()
    encoder.load_state_dict(encoder_state)
    return encoder


class EnhancerClassifier(L.LightningModule):
    """Binary enhancer classifier using AlphaGenome's CNN encoder trunk.

    Architecture: SequenceEncoder → AdaptiveAvgPool1d(1) → Linear(1536, 1)

    Args:
        weights_path: Path to AlphaGenome pretrained weights. If None, the
            encoder is randomly initialized (useful for tests and checkpoint
            restore).
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        freeze_backbone: If True, freeze all encoder parameters.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        freeze_backbone: bool = True,
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

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ENCODER_OUTPUT_DIM, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk, _intermediates = self.encoder(x)
        logits = self.head(trunk).squeeze(-1)
        return logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auroc.update(preds, y.int())
        self.val_auprc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

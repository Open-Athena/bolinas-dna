"""LoRA fine-tuning for VEP with a pair-aware ranking loss on `embed_last_l2`.

Iter-2 of #180. The architecture is deliberately minimal:

* **Score** = flattened-sequence pairwise L2 distance between the last hidden
  states of the ref and alt contexts: ``||flat(last_ref) − flat(last_alt)||₂``.
  Identical to the frozen-embedding zero-shot baseline (`embed_last_l2`)
  identified as best across all 3 datasets in iter-1b.
* **Only learnable parameters** = LoRA adapters on the backbone (q_proj,
  v_proj by default). No classification head, no MLP. At LoRA initialization
  (zero-rank adapters) the score is identical to the frozen baseline, so the
  training only has work to do where the baseline is sub-optimal.
* **Loss** = pair-aware ranking loss
  ``softplus(margin − (score_pos − score_neg))`` over matched (pos, neg)
  pairs — directly optimises within-`match_group` PairwiseAccuracy, the eval
  metric.
* The L2 distance is symmetric in ref↔alt by construction → same architecture
  for all 3 datasets, no mendelian-asymmetric variant needed.

The training loop is plain torch — the pair-aware loss + paired dataloader
don't fit cleanly inside HF Trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from biofoundation.data import Genome, transform_llr_clm
from biofoundation.model.adapters.hf import HFTokenizer
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoraConfigSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class PairwiseVepLora(nn.Module):
    """LoRA-adapted CausalLM scored via flattened-sequence L2 distance.

    Forward signature: ``input_ids`` of shape ``[B, 2, L]`` where dim 1 is
    ``(ref_context, alt_context)``. Returns ``[B]`` scalar score per variant
    equal to ``||flat(last_ref) − flat(last_alt)||₂``.
    """

    def __init__(
        self,
        backbone_id: str,
        lora: LoraConfigSpec = LoraConfigSpec(),
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
        normalize: bool = False,
        use_head: bool = False,
        head_hidden_div: int = 4,
        head_dropout: float = 0.1,
    ) -> None:
        """Two scoring architectures controlled by ``use_head``:

        ``use_head=False`` (default): score = pairwise L2 distance over the
        flattened ``L*D`` hidden state. With ``normalize=True``, the flattened
        vector is L2-normalized to the unit sphere first → score ∈ [0, 2].

        ``use_head=True``: mean-pool last hidden state, then score = MLP head
        on the symmetric feature ``|alt_pool − ref_pool|``. The MLP is
        ``D → D/head_hidden_div → 1`` with GELU + dropout. LoRA-init no longer
        matches the frozen ``embed_last_l2`` baseline (head is random at init).
        """
        super().__init__()
        backbone = AutoModelForCausalLM.from_pretrained(
            backbone_id, trust_remote_code=True, torch_dtype=torch_dtype
        )
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
        peft_config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            target_modules=list(lora.target_modules),
            lora_dropout=lora.dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(backbone, peft_config)
        self.normalize = normalize
        self.use_head = use_head
        if use_head:
            D = backbone.config.hidden_size
            self.head = nn.Sequential(
                nn.Linear(D, max(D // head_hidden_div, 16)),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(max(D // head_hidden_div, 16), 1),
            )
        else:
            self.head = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B = input_ids.shape[0]
        flat = rearrange(input_ids, "B V L -> (B V) L")
        out = self.backbone(input_ids=flat, output_hidden_states=True)
        last = out.hidden_states[-1]  # [B*2, L, D]

        if self.use_head:
            # Mean-pool over positions → [B*2, D], then symmetric |alt - ref| → MLP → scalar.
            pooled = last.mean(dim=1)  # [B*2, D]
            pooled = rearrange(pooled, "(B V) D -> B V D", B=B)
            feat = (pooled[:, 1] - pooled[:, 0]).abs().float()  # [B, D] in fp32 for the head
            return self.head(feat).squeeze(-1)

        # No-head: flat L2 distance (optionally normalized).
        last = rearrange(last, "(B V) L D -> B V (L D)", B=B)  # flatten L,D
        if self.normalize:
            last = F.normalize(last, p=2, dim=-1)
        return F.pairwise_distance(last[:, 0], last[:, 1])  # [B]

    def trainable_parameters(self) -> int:
        """Count LoRA-adapter params (everything else should be frozen)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------- loss ----------------------------------------------------------


def pairwise_ranking_loss(
    score_pos: torch.Tensor,
    score_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """RankNet-style ranking loss: ``softplus(margin − (s_pos − s_neg))``."""
    return F.softplus(margin - (score_pos - score_neg)).mean()


# ---------- datasets ------------------------------------------------------


def _tokenize_variant(
    row: pd.Series, tokenizer: HFTokenizer, genome: Genome, window_size: int
) -> torch.Tensor:
    """Re-use ``transform_llr_clm`` for the [2, L] (ref, alt) token pair."""
    example = {
        "chrom": row["chrom"],
        "pos": int(row["pos"]),
        "ref": str(row["ref"]),
        "alt": str(row["alt"]),
    }
    out = transform_llr_clm(
        example, tokenizer=tokenizer, genome=genome, window_size=window_size
    )
    return out["input_ids"]


class PairedVariantDataset(Dataset):
    """One sample per ``match_group`` = (pos_variant, neg_variant) pair.

    Groups that don't have exactly one positive and one negative are skipped
    silently (caller can detect via len()).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: HFTokenizer,
        genome: Genome,
        window_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.genome = genome
        self.window_size = window_size
        pairs: list[tuple[pd.Series, pd.Series]] = []
        for _mg, group in df.groupby("match_group"):
            pos = group[group["label"].astype(bool)]
            neg = group[~group["label"].astype(bool)]
            if len(pos) != 1 or len(neg) != 1:
                continue
            pairs.append((pos.iloc[0], neg.iloc[0]))
        if not pairs:
            raise ValueError("PairedVariantDataset got 0 valid pairs")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pos_row, neg_row = self.pairs[idx]
        return {
            "pos_input_ids": _tokenize_variant(
                pos_row, self.tokenizer, self.genome, self.window_size
            ),
            "neg_input_ids": _tokenize_variant(
                neg_row, self.tokenizer, self.genome, self.window_size
            ),
        }


class VariantDataset(Dataset):
    """One sample per variant — used for OOF scoring after training."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: HFTokenizer,
        genome: Genome,
        window_size: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.genome = genome
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _tokenize_variant(
            self.df.iloc[idx], self.tokenizer, self.genome, self.window_size
        )


# ---------- training / prediction ----------------------------------------


def train_one_fold(
    model: PairwiseVepLora,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    tokenizer: HFTokenizer,
    genome: Genome,
    window_size: int,
    *,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    margin: float = 1.0,
    num_workers: int = 2,
    device: str = "cuda",
    log_every: int = 20,
) -> dict[str, Any]:
    """Train LoRA adapters with pair-aware ranking loss; track overfitting.

    Per-epoch eval on both ``train_df`` (to detect overfitting via train PA →
    1.0 while val PA stalls) and ``val_df``. Also reports epoch-0 numbers
    (before any training) so the lift over the frozen-baseline score is
    visible directly.

    No early-stopping: we run the full ``epochs`` budget and return the curve
    so the caller can inspect overfitting and choose the right epoch count
    by hand. Auto-early-stop on val would turn val into a tunable hparam and
    contaminate the outer-test fold's protection.
    """
    train_ds = PairedVariantDataset(train_df, tokenizer, genome, window_size)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)
    n_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, total_steps=n_steps, pct_start=0.1
    )

    model.to(device)
    stats: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_pa": [],
        "val_pa": [],
    }

    # Epoch-0 baseline (LoRA adapters at init = identity = frozen score).
    train_pa_0 = evaluate_pa(
        model,
        train_df,
        tokenizer,
        genome,
        window_size,
        device=device,
        batch_size=batch_size,
    )
    val_pa_0 = None
    if val_df is not None:
        val_pa_0 = evaluate_pa(
            model,
            val_df,
            tokenizer,
            genome,
            window_size,
            device=device,
            batch_size=batch_size,
        )
    stats["epoch"].append(0)
    stats["train_loss"].append(None)
    stats["train_pa"].append(train_pa_0)
    stats["val_pa"].append(val_pa_0)
    print(
        f"epoch 0 (frozen baseline)  train_PA={train_pa_0:.4f}"
        + (f"  val_PA={val_pa_0:.4f}" if val_pa_0 is not None else "")
    )

    step = 0
    optim.zero_grad()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        accum_count = 0
        for batch in loader:
            pos_ids = batch["pos_input_ids"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            s_pos = model(pos_ids)
            s_neg = model(neg_ids)
            loss = pairwise_ranking_loss(s_pos, s_neg, margin=margin)
            (loss / grad_accum_steps).backward()
            accum_count += 1
            if accum_count >= grad_accum_steps:
                optim.step()
                sched.step()
                optim.zero_grad()
                accum_count = 0
            loss_sum += loss.item()
            n_batches += 1
            step += 1
            if step % log_every == 0:
                print(
                    f"  step {step}/{n_steps}  loss={loss.item():.4f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )
        # Flush any partial accumulation at end of epoch.
        if accum_count > 0:
            optim.step()
            sched.step()
            optim.zero_grad()
        train_loss = loss_sum / max(n_batches, 1)
        train_pa = evaluate_pa(
            model,
            train_df,
            tokenizer,
            genome,
            window_size,
            device=device,
            batch_size=batch_size,
        )
        val_pa = None
        if val_df is not None:
            val_pa = evaluate_pa(
                model,
                val_df,
                tokenizer,
                genome,
                window_size,
                device=device,
                batch_size=batch_size,
            )
        stats["epoch"].append(epoch)
        stats["train_loss"].append(train_loss)
        stats["train_pa"].append(train_pa)
        stats["val_pa"].append(val_pa)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"train_PA={train_pa:.4f}"
            + (f"  val_PA={val_pa:.4f}" if val_pa is not None else "")
        )
    return stats


@torch.no_grad()
def predict_scores(
    model: PairwiseVepLora,
    df: pd.DataFrame,
    tokenizer: HFTokenizer,
    genome: Genome,
    window_size: int,
    *,
    batch_size: int = 16,
    num_workers: int = 2,
    device: str = "cuda",
) -> np.ndarray:
    """Return per-variant scores aligned with ``df.reset_index(drop=True)``."""
    ds = VariantDataset(df, tokenizer, genome, window_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    for input_ids in loader:
        input_ids = input_ids.to(device)
        scores = model(input_ids).float().cpu().numpy()
        out.append(scores)
    return np.concatenate(out)


@torch.no_grad()
def evaluate_pa(
    model: PairwiseVepLora,
    df: pd.DataFrame,
    tokenizer: HFTokenizer,
    genome: Genome,
    window_size: int,
    *,
    batch_size: int = 16,
    device: str = "cuda",
) -> float:
    """Within-``match_group`` PairwiseAccuracy on the given dataframe."""
    scores = predict_scores(
        model,
        df,
        tokenizer,
        genome,
        window_size,
        batch_size=batch_size,
        device=device,
    )
    df = df.reset_index(drop=True).copy()
    df["_score"] = scores
    correct = 0.0
    total = 0
    for _mg, g in df.groupby("match_group"):
        if len(g) != 2:
            continue
        labels = g["label"].astype(int).to_numpy()
        if labels.sum() != 1:
            continue
        s = g["_score"].to_numpy()
        s_pos = s[labels == 1][0]
        s_neg = s[labels == 0][0]
        if s_pos > s_neg:
            correct += 1
        elif s_pos == s_neg:
            correct += 0.5
        total += 1
    return correct / max(total, 1)


def load_tokenizer_and_genome(
    backbone_id: str, genome_path: str | Path
) -> tuple[HFTokenizer, Genome]:
    tok = HFTokenizer(AutoTokenizer.from_pretrained(backbone_id))
    return tok, Genome(Path(genome_path))

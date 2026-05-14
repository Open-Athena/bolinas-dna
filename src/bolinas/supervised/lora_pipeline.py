"""High-level per-fold LoRA train + predict entry point for the snakemake rule.

Keeps the Snakemake ``run:`` block thin (per CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from bolinas.supervised.cv import assign_chrom_folds
from bolinas.supervised.lora import (
    LoraConfigSpec,
    PairwiseVepLora,
    load_tokenizer_and_genome,
    predict_scores,
    train_one_fold,
)


META_COLS = ("chrom", "pos", "ref", "alt", "label", "subset", "match_group")


def fit_predict_one_fold(
    *,
    hf_dataset_path: str,
    split: str,
    backbone_id: str,
    window_size: int,
    genome_path: str | Path,
    fold: int,
    n_splits: int = 3,
    lora_rank: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    normalize: bool = True,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 2,
    margin: float = 0.5,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[pd.DataFrame, dict]:
    """Train LoRA on (n_splits-1) chrom-folds, predict on the held-out fold.

    Returns ``(predictions_df, training_stats)``.

    The predictions df has the META_COLS + ``score`` + ``fold`` + ``family``
    columns, aligned with the held-out variants (one row per variant in that
    fold's test chromosomes).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_dataset(hf_dataset_path, split=split).to_pandas()
    for col in META_COLS:
        assert col in df.columns, f"dataset missing column {col!r}"

    chroms = df["chrom"].astype(str).to_numpy()
    folds = assign_chrom_folds(chroms, n_splits=n_splits)
    fold_per_row = np.array([folds.fold_index(c) for c in chroms])
    test_mask = fold_per_row == fold
    test_df = df.loc[test_mask].reset_index(drop=True)
    outer_train_chroms = sorted(set(chroms[~test_mask]))
    test_chroms = sorted(set(chroms[test_mask]))

    # Inner val split: pull one chrom from the outer-train set as a val fold
    # used for diagnostics + overfitting curves only. The outer-test fold is
    # NEVER touched during training. Deterministic pick — alphabetically
    # first chrom in the outer-train set, sorted as strings (matches the
    # ordering inside ``assign_chrom_folds``).
    inner_val_chrom = outer_train_chroms[0]
    inner_val_mask = df["chrom"].astype(str) == inner_val_chrom
    inner_train_df = df.loc[~test_mask & ~inner_val_mask].reset_index(drop=True)
    inner_val_df = df.loc[~test_mask & inner_val_mask].reset_index(drop=True)
    inner_train_chroms = sorted(set(inner_train_df["chrom"].astype(str)))
    print(
        f"[fit_predict_one_fold] fold={fold}/{n_splits}: "
        f"inner_train n={len(inner_train_df)} chroms={inner_train_chroms} | "
        f"inner_val n={len(inner_val_df)} chrom={inner_val_chrom} | "
        f"outer_test n={len(test_df)} chroms={test_chroms}"
    )

    tokenizer, genome = load_tokenizer_and_genome(backbone_id, genome_path)
    model = PairwiseVepLora(
        backbone_id=backbone_id,
        lora=LoraConfigSpec(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=tuple(lora_target_modules),
        ),
        normalize=normalize,
    )
    n_trainable = model.trainable_parameters()
    print(f"[fit_predict_one_fold] LoRA trainable params: {n_trainable:,}")

    stats = train_one_fold(
        model,
        train_df=inner_train_df,
        val_df=inner_val_df,  # held-out chrom from outer-train; outer-test never seen
        tokenizer=tokenizer,
        genome=genome,
        window_size=window_size,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        margin=margin,
        num_workers=num_workers,
        device=device,
    )

    scores = predict_scores(
        model,
        test_df,
        tokenizer=tokenizer,
        genome=genome,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    preds = test_df[list(META_COLS)].copy()
    preds["score"] = scores
    preds["fold"] = fold
    preds["family"] = "lora"
    stats["fold"] = fold
    stats["n_trainable_params"] = n_trainable
    stats["inner_train_chroms"] = inner_train_chroms
    stats["inner_val_chrom"] = inner_val_chrom
    stats["outer_test_chroms"] = test_chroms
    return preds, stats

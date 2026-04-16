"""Whole-genome enhancer prediction with per-bin segmentation model.

Tiles genomic regions with non-overlapping windows and runs the
EnhancerSegmenter to produce per-bin logits at 128bp resolution.

Usage::

    python -m bolinas.enhancer_segmentation.predict_genome \
        --genome genome.2bit \
        --checkpoint best.ckpt \
        --windows windows.parquet \
        --output predictions.parquet
"""

import argparse
import logging
import time
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from torch.utils.data import DataLoader, Dataset

from bolinas.enhancer_classification.genome import Genome
from bolinas.enhancer_segmentation.model import EnhancerSegmenter

logger = logging.getLogger(__name__)


class GenomeWindowDataset(Dataset):
    """Map-style dataset of genomic windows for segmentation inference.

    Window coordinates come from a pre-computed parquet file. Sequences are
    lazily extracted from the genome in ``__getitem__``, so each DataLoader
    worker opens its own file handle (shared memory-mapped pages for 2bit).

    Args:
        genome_path: Path to genome file (.2bit or .fa/.fa.gz).
        windows: DataFrame with ``chrom``, ``start``, ``end`` columns.
    """

    def __init__(self, genome_path: Path, windows: pl.DataFrame) -> None:
        self._genome_path = genome_path
        self._chroms = windows["chrom"].to_numpy()
        self._starts = windows["start"].to_numpy().astype(np.int64)
        self._ends = windows["end"].to_numpy().astype(np.int64)
        self._genome: Genome | None = None

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._genome is None:
            self._genome = Genome(self._genome_path)
        seq = self._genome(
            str(self._chroms[idx]), int(self._starts[idx]), int(self._ends[idx])
        )
        onehot = sequence_to_onehot(seq).astype(np.float32)
        return torch.from_numpy(onehot)


def predict_genome(
    genome_path: Path,
    checkpoint_path: Path,
    windows: pl.DataFrame,
    bin_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
) -> pl.DataFrame:
    """Run segmentation prediction on pre-computed windows.

    Args:
        genome_path: Path to genome file (.2bit or .fa/.fa.gz).
        checkpoint_path: Path to trained EnhancerSegmenter checkpoint.
        windows: DataFrame with ``chrom``, ``start``, ``end`` columns.
        bin_size: Size of each output bin in bp.
        batch_size: Inference batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        DataFrame with columns ``chrom``, ``bin_start``, ``bin_end``, ``logit``
        — one row per bin across all windows.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    model = EnhancerSegmenter.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    model = torch.compile(model)

    dataset = GenomeWindowDataset(genome_path, windows)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    trainer = L.Trainer(
        accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
    )

    t0 = time.perf_counter()
    predictions = trainer.predict(model, dataloader)
    elapsed = time.perf_counter() - t0

    assert predictions is not None
    all_logits = np.concatenate([p.float().numpy() for p in predictions])
    num_bins = all_logits.shape[1]
    total_bins = len(windows) * num_bins

    logger.info(
        "Predicted %d windows (%d bins) in %.1fs (%.0f bins/sec)",
        len(windows),
        total_bins,
        elapsed,
        total_bins / elapsed,
    )

    chroms = np.repeat(windows["chrom"].to_numpy(), num_bins)
    window_starts = np.repeat(windows["start"].to_numpy(), num_bins)
    bin_indices = np.tile(np.arange(num_bins, dtype=np.int32), len(windows))
    bin_starts = window_starts + bin_indices * bin_size
    bin_ends = bin_starts + bin_size

    return pl.DataFrame(
        {
            "chrom": chroms,
            "bin_start": bin_starts,
            "bin_end": bin_ends,
            "logit": all_logits.reshape(-1).astype(np.float32),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-bin enhancer logits across genomic regions."
    )
    parser.add_argument(
        "--genome", type=Path, required=True, help="Genome file (.2bit or .fa.gz)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Lightning checkpoint"
    )
    parser.add_argument(
        "--windows", type=Path, required=True, help="Window coordinates parquet"
    )
    parser.add_argument("--bin-size", type=int, default=128, help="Bin size in bp")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Truncate windows to first N for smoke tests (0 = no limit)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading windows from %s", args.windows)
    windows = pl.read_parquet(args.windows)
    if args.max_windows > 0:
        windows = windows.head(args.max_windows)
        logger.info("  Truncated to first %d windows for smoke test", args.max_windows)
    logger.info(
        "  %d windows across %d chromosomes", len(windows), windows["chrom"].n_unique()
    )

    result = predict_genome(
        genome_path=args.genome,
        checkpoint_path=args.checkpoint,
        windows=windows,
        bin_size=args.bin_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output)
    logger.info("Wrote %s (%d bins)", args.output, len(result))


if __name__ == "__main__":
    main()

"""Evaluate a trained EnhancerSegmenter checkpoint on a validation parquet.

Runs a single forward pass over the entire validation parquet (no training,
no gradient) and writes a per-(window, bin) val_predictions parquet compatible
with the AUPRC / precision-recall tooling. Used to re-evaluate existing
checkpoints against a freshly built full-coverage validation set.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from bolinas.enhancer_segmentation.dataset import SegmentationDataset
from bolinas.enhancer_segmentation.model import EnhancerSegmenter

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a segmenter checkpoint on a val parquet"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val-parquet", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnhancerSegmenter.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    genome_to_idx = (
        {g: i for i, g in enumerate(model.genomes)} if model.genomes else None
    )
    val_ds = SegmentationDataset(
        args.val_parquet, augment_rc=False, genome_to_idx=genome_to_idx
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x, _y, _g = batch
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)  # (N, num_bins)
    num_bins = logits_arr.shape[1]
    log.info("Ran inference on %d windows x %d bins", logits_arr.shape[0], num_bins)

    val_meta = pl.read_parquet(
        args.val_parquet,
        columns=["genome", "chrom", "start", "end", "strand", "labels"],
    )
    assert len(val_meta) == logits_arr.shape[0], (
        f"Val rows {len(val_meta)} != logits rows {logits_arr.shape[0]}"
    )

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
            "bin_start": window_starts + bin_idx * bin_size,
            "bin_end": window_starts + (bin_idx + 1) * bin_size,
            "label": labels_flat,
            "logit": logits_arr.reshape(-1),
        }
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(args.output)
    log.info("Wrote %d prediction rows to %s", predictions.height, args.output)


if __name__ == "__main__":
    main()

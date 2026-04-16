"""One-off benchmark: segmentation prediction on a single human chromosome.

Tiles one chromosome with non-overlapping 64kbp windows, runs the
segmentation model, and reports wall time + throughput.

Usage::

    uv run python scripts/benchmark_segmentation_predict.py \
        --genome results/genome/GCF_000001405.40.2bit \
        --checkpoint results/segmentation/model/xfmr2_w64k_s42_full/seg_v1_64k/best.ckpt \
        --chrom 22 \
        --output /tmp/benchmark_chr22.parquet
"""

import argparse
import logging
from pathlib import Path

import polars as pl
import py2bit

from bolinas.enhancer_segmentation.predict_genome import predict_genome

logger = logging.getLogger(__name__)

WINDOW_SIZE = 65536
BIN_SIZE = 128


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Benchmark segmentation on one chromosome."
    )
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chrom", type=str, default="22")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tb = py2bit.open(str(args.genome))
    chrom_size = tb.chroms()[args.chrom]
    tb.close()

    starts = list(range(0, chrom_size, WINDOW_SIZE))
    windows = pl.DataFrame(
        {
            "chrom": [args.chrom] * len(starts),
            "start": starts,
            "end": [s + WINDOW_SIZE for s in starts],
        }
    )

    logger.info(
        "Chromosome %s: %d bp, %d windows (%d bins)",
        args.chrom,
        chrom_size,
        len(windows),
        len(windows) * (WINDOW_SIZE // BIN_SIZE),
    )

    result = predict_genome(
        genome_path=args.genome,
        checkpoint_path=args.checkpoint,
        windows=windows,
        bin_size=BIN_SIZE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(args.output)
        logger.info("Wrote %s", args.output)

    logger.info("Logit stats: %s", result["logit"].describe())


if __name__ == "__main__":
    main()

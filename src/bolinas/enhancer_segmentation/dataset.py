"""PyTorch dataset for per-bin enhancer segmentation."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from Bio.Seq import Seq
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Per-bin segmentation dataset backed by a parquet file.

    Each item is a ``(one_hot, labels)`` pair where ``one_hot`` is a
    float32 tensor of shape ``(window_size, 4)`` and ``labels`` is a
    float32 tensor of shape ``(num_bins,)``.

    With ``augment_rc=True`` the dataset's virtual length is doubled: index
    ``i < N`` returns the forward sample, index ``i >= N`` returns the
    reverse-complement of sample ``i - N`` with its per-bin labels reversed.
    This matches pre-baking ``forward + RC`` into the parquet but avoids the
    2x memory spike at dataset-build time.

    Args:
        parquet_path: Path to a parquet file with columns ``seq`` (str) and
            ``labels`` (list[int] / list[uint8]).
        augment_rc: If True (train split), double virtual length with RC
            samples. Default False (val split).
    """

    def __init__(self, parquet_path: str | Path, augment_rc: bool = False) -> None:
        df = pl.read_parquet(parquet_path, columns=["seq", "labels"])
        self.sequences: list[str] = df["seq"].to_list()
        # Store labels as a single (N, num_bins) uint8 array for fast indexing.
        self.labels: np.ndarray = np.asarray(df["labels"].to_list(), dtype=np.uint8)
        self.augment_rc = augment_rc
        self._n = len(self.sequences)

    def __len__(self) -> int:
        return 2 * self._n if self.augment_rc else self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment_rc and idx >= self._n:
            base = idx - self._n
            seq = str(Seq(self.sequences[base]).reverse_complement())
            labels = self.labels[base][::-1].astype(np.float32)
        else:
            seq = self.sequences[idx]
            labels = self.labels[idx].astype(np.float32)
        onehot = sequence_to_onehot(seq).astype(np.float32)
        return (
            torch.from_numpy(onehot),
            torch.from_numpy(np.ascontiguousarray(labels)),
        )

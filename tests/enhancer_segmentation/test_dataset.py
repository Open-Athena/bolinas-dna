"""Tests for SegmentationDataset, especially on-the-fly RC augmentation."""

import polars as pl
import torch
from Bio.Seq import Seq

from bolinas.enhancer_segmentation.dataset import SegmentationDataset

NUM_BINS = 4
SEQ = "ACGT" * 8  # 32 bp; NUM_BINS=4 means bin_size=8, only used for test shape


def _write_parquet(tmp_path):
    # Three windows; labels = three bin-label patterns that survive reversal
    # visibly: [1, 0, 0, 0] reversed = [0, 0, 0, 1], etc.
    df = pl.DataFrame(
        {
            "seq": [SEQ, SEQ, SEQ],
            "labels": [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
        },
        schema={"seq": pl.Utf8, "labels": pl.List(pl.UInt8)},
    )
    path = tmp_path / "ds.parquet"
    df.write_parquet(path)
    return path


def test_no_rc_yields_n_samples(tmp_path):
    ds = SegmentationDataset(_write_parquet(tmp_path), augment_rc=False)
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (len(SEQ), 4)
    assert y.shape == (NUM_BINS,)
    assert torch.equal(y, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_rc_doubles_length(tmp_path):
    ds = SegmentationDataset(_write_parquet(tmp_path), augment_rc=True)
    assert len(ds) == 6


def test_rc_reverses_labels_and_rev_complements_seq(tmp_path):
    ds = SegmentationDataset(_write_parquet(tmp_path), augment_rc=True)

    # Forward sample 0
    x_fwd, y_fwd = ds[0]
    assert torch.equal(y_fwd, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # RC sample (fwd index 0 -> rc index 3)
    x_rc, y_rc = ds[3]
    assert torch.equal(y_rc, torch.tensor([0.0, 0.0, 0.0, 1.0]))

    # The one-hot of the RC sample should equal the one-hot of the RC string.
    from alphagenome_pytorch.utils.sequence import sequence_to_onehot

    expected_rc_seq = str(Seq(SEQ).reverse_complement())
    expected_onehot = torch.from_numpy(
        sequence_to_onehot(expected_rc_seq).astype("float32")
    )
    assert torch.equal(x_rc, expected_onehot)


def test_rc_preserves_palindrome_invariance(tmp_path):
    """For a symmetric label pattern, fwd and rc labels should match."""
    ds = SegmentationDataset(_write_parquet(tmp_path), augment_rc=True)
    # Sample index 1: labels = [0, 1, 1, 0] — palindrome, so RC label matches.
    _x_fwd, y_fwd = ds[1]
    _x_rc, y_rc = ds[1 + 3]
    assert torch.equal(y_fwd, y_rc)

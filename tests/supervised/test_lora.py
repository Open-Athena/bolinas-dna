"""Tests for ``bolinas.supervised.lora`` — pair-aware ranking loss + dataset.

We don't load the real 1B backbone in tests (too slow); just exercise the
loss function and dataset construction with synthetic data. The full model
forward gets exercised on a tiny HF model in a separate slow test, gated
behind a marker.
"""

import pandas as pd
import pytest
import torch

from bolinas.supervised.lora import (
    PairedVariantDataset,
    pairwise_ranking_loss,
)


# ---------- loss tests ----------------------------------------------------


def test_pairwise_ranking_loss_is_zero_when_score_diff_exceeds_margin():
    s_pos = torch.tensor([5.0, 7.0])
    s_neg = torch.tensor([0.0, 1.0])
    # softplus(1 - (5 - 0)) = softplus(-4) ≈ 0.018; softplus(1 - (7 - 1)) = softplus(-5) ≈ 0.007
    loss = pairwise_ranking_loss(s_pos, s_neg, margin=1.0).item()
    assert 0.0 < loss < 0.02


def test_pairwise_ranking_loss_grows_when_score_diff_is_wrong_direction():
    s_pos = torch.tensor([-2.0])
    s_neg = torch.tensor([2.0])
    # softplus(1 - (-2 - 2)) = softplus(5) ≈ 5.007
    loss = pairwise_ranking_loss(s_pos, s_neg, margin=1.0).item()
    assert loss > 4.9


def test_pairwise_ranking_loss_handles_batch():
    rng = torch.Generator().manual_seed(0)
    s_pos = torch.randn(32, generator=rng)
    s_neg = torch.randn(32, generator=rng)
    loss = pairwise_ranking_loss(s_pos, s_neg)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_pairwise_ranking_loss_is_differentiable():
    s_pos = torch.zeros(4, requires_grad=True)
    s_neg = torch.zeros(4)
    loss = pairwise_ranking_loss(s_pos, s_neg)
    loss.backward()
    # At s_pos = s_neg, the gradient direction should push s_pos up.
    assert (
        s_pos.grad < 0
    ).all()  # gradient on softplus(1 - 0) at the point pushes s_pos upward


# ---------- dataset tests -------------------------------------------------


class _StubTokenizer:
    """Returns a fixed-length token vector for any sequence."""

    def encode(self, seq: str) -> list[int]:
        return [hash(c) & 0xFF for c in seq[:32].ljust(32, "N")]


class _StubGenome:
    """Callable genome stub. Always returns the variant's ref base at the
    centre position (to pass the assertion inside ``transform_llr_clm``);
    everything else is filler 'A'."""

    def __init__(self, ref_by_pos: dict[tuple[str, int], str]) -> None:
        self.ref_by_pos = ref_by_pos

    def __call__(self, chrom: str, start: int, end: int) -> str:
        window_size = end - start
        center = start + window_size // 2
        # 1-based pos in ref_by_pos.
        ref = self.ref_by_pos.get((chrom, center + 1), "A")
        seq = ["A"] * window_size
        seq[window_size // 2] = ref
        return "".join(seq)


_COLS = ["chrom", "pos", "ref", "alt", "label", "match_group"]


def _make_paired_df(n_pairs: int = 5) -> tuple[pd.DataFrame, _StubGenome]:
    rows = []
    ref_by_pos: dict[tuple[str, int], str] = {}
    for i in range(n_pairs):
        ppos = 1000 + i * 10
        npos = 1000 + i * 10 + 5
        rows.append(
            {
                "chrom": "1",
                "pos": ppos,
                "ref": "A",
                "alt": "G",
                "label": True,
                "match_group": i,
            }
        )
        rows.append(
            {
                "chrom": "1",
                "pos": npos,
                "ref": "C",
                "alt": "T",
                "label": False,
                "match_group": i,
            }
        )
        ref_by_pos[("1", ppos)] = "A"
        ref_by_pos[("1", npos)] = "C"
    df = pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    return df, _StubGenome(ref_by_pos)


def test_paired_dataset_returns_one_sample_per_match_group():
    df, genome = _make_paired_df(n_pairs=7)
    tok = _StubTokenizer()
    ds = PairedVariantDataset(df, tok, genome, window_size=16)
    assert len(ds) == 7
    sample = ds[0]
    assert set(sample.keys()) == {"pos_input_ids", "neg_input_ids"}
    assert sample["pos_input_ids"].dim() == 2  # [2, L]
    assert sample["pos_input_ids"].shape[0] == 2  # ref + alt


def test_paired_dataset_skips_groups_without_one_pos_one_neg():
    df, genome = _make_paired_df(n_pairs=3)
    # Add a degenerate group with two positives.
    extra = pd.DataFrame(
        [
            {
                "chrom": "1",
                "pos": 9000,
                "ref": "A",
                "alt": "G",
                "label": True,
                "match_group": 99,
            },
            {
                "chrom": "1",
                "pos": 9001,
                "ref": "A",
                "alt": "G",
                "label": True,
                "match_group": 99,
            },
        ]
    )
    df = pd.concat([df, extra], ignore_index=True)
    ds = PairedVariantDataset(df, _StubTokenizer(), genome, window_size=16)
    # Three valid pairs, the degenerate group is skipped.
    assert len(ds) == 3


def test_paired_dataset_rejects_empty():
    df, genome = _make_paired_df(n_pairs=0)
    with pytest.raises(ValueError, match="0 valid pairs"):
        PairedVariantDataset(df, _StubTokenizer(), genome, window_size=16)


def test_paired_dataset_loadable_via_dataloader():
    from torch.utils.data import DataLoader

    df, genome = _make_paired_df(n_pairs=6)
    ds = PairedVariantDataset(df, _StubTokenizer(), genome, window_size=16)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch["pos_input_ids"].shape == (3, 2, 32)
    assert batch["neg_input_ids"].shape == (3, 2, 32)

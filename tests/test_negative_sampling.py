import numpy as np
import pandas as pd

from bolinas.data.negative_sampling import (
    compute_gc_content,
    compute_repeat_fraction,
    match_by_gc_repeat,
)


def test_compute_gc_content_all_gc():
    seqs = pd.Series(["GCGCGC", "GGGGGG"])
    result = compute_gc_content(seqs)
    assert result.tolist() == [1.0, 1.0]


def test_compute_gc_content_no_gc():
    seqs = pd.Series(["ATATAT", "TTTTTT"])
    result = compute_gc_content(seqs)
    assert result.tolist() == [0.0, 0.0]


def test_compute_gc_content_mixed():
    seqs = pd.Series(["ATGC"])  # 2 GC out of 4
    result = compute_gc_content(seqs)
    assert result.iloc[0] == 0.5


def test_compute_gc_content_case_insensitive():
    seqs = pd.Series(["atgc", "ATGC", "AtGc"])
    result = compute_gc_content(seqs)
    np.testing.assert_array_equal(result.to_numpy(), [0.5, 0.5, 0.5])


def test_compute_repeat_fraction_all_lowercase():
    seqs = pd.Series(["atgcatgc"])
    result = compute_repeat_fraction(seqs)
    assert result.iloc[0] == 1.0


def test_compute_repeat_fraction_no_lowercase():
    seqs = pd.Series(["ATGCATGC"])
    result = compute_repeat_fraction(seqs)
    assert result.iloc[0] == 0.0


def test_compute_repeat_fraction_mixed():
    seqs = pd.Series(["ATGCatgc"])  # 4 lowercase out of 8
    result = compute_repeat_fraction(seqs)
    assert result.iloc[0] == 0.5


def test_match_by_gc_repeat_basic():
    """Matched negatives should have similar GC/repeat to positives."""
    rng = np.random.default_rng(42)
    n_pos = 100
    n_cand = 2000

    # Positives: GC ~0.5, repeat ~0.3
    pos_seqs = pd.Series(
        [_make_seq(rng, gc=0.5, repeat=0.3, length=100) for _ in range(n_pos)]
    )
    # Candidates: wide range of GC and repeat
    cand_seqs = pd.Series(
        [
            _make_seq(rng, gc=rng.uniform(0.2, 0.8), repeat=rng.uniform(0.0, 0.8), length=100)
            for _ in range(n_cand)
        ]
    )

    indices = match_by_gc_repeat(pos_seqs, cand_seqs, seed=42)

    assert len(indices) == n_pos
    assert len(set(indices)) == n_pos  # all unique

    # Matched negatives should be closer in GC to positives than random sample
    pos_gc = compute_gc_content(pos_seqs).to_numpy()
    matched_gc = compute_gc_content(cand_seqs.iloc[indices]).to_numpy()
    random_gc = compute_gc_content(cand_seqs.iloc[:n_pos]).to_numpy()

    matched_diff = np.abs(pos_gc - matched_gc).mean()
    random_diff = np.abs(pos_gc - random_gc).mean()
    assert matched_diff < random_diff


def test_match_by_gc_repeat_all_matched_via_bins():
    """When candidates span the same bins as positives, all should bin-match."""
    rng = np.random.default_rng(0)
    n_pos = 50
    n_cand = 1000

    # Both positives and candidates in a narrow GC/repeat range
    pos_seqs = pd.Series(
        [_make_seq(rng, gc=0.45, repeat=0.2, length=100) for _ in range(n_pos)]
    )
    cand_seqs = pd.Series(
        [_make_seq(rng, gc=0.45, repeat=0.2, length=100) for _ in range(n_cand)]
    )

    indices = match_by_gc_repeat(pos_seqs, cand_seqs, seed=0)
    assert len(indices) == n_pos
    assert len(set(indices)) == n_pos


def _make_seq(rng: np.random.Generator, gc: float, repeat: float, length: int) -> str:
    """Generate a synthetic DNA sequence with approximate GC and repeat fraction."""
    bases_upper = []
    for _ in range(length):
        if rng.random() < gc:
            bases_upper.append(rng.choice(["G", "C"]))
        else:
            bases_upper.append(rng.choice(["A", "T"]))

    # Apply repeat masking (lowercase)
    chars = list(bases_upper)
    for i in range(length):
        if rng.random() < repeat:
            chars[i] = chars[i].lower()

    return "".join(chars)

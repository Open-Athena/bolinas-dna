"""GC and repeat-content matched negative sampling for enhancer classification.

Matches negative (non-enhancer) genomic regions to positive (enhancer) regions
based on GC content and repeat fraction, removing compositional confounds that
hinder cross-species generalization.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def compute_gc_content(sequences: pd.Series) -> pd.Series:
    """Fraction of G+C bases (case-insensitive) per sequence."""
    gc_count = sequences.str.count("[GCgc]")
    return gc_count / sequences.str.len()


def compute_repeat_fraction(sequences: pd.Series) -> pd.Series:
    """Fraction of lowercase (repeat-masked) bases in soft-masked sequences."""
    lowercase_count = sequences.str.count("[acgt]")
    return lowercase_count / sequences.str.len()


def match_by_gc_repeat(
    positive_seqs: pd.Series,
    candidate_seqs: pd.Series,
    gc_bin_size: float = 0.02,
    repeat_bin_size: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Match each positive to a candidate with similar GC and repeat content.

    Uses bin-based matching: positives and candidates are assigned to 2D bins
    defined by (GC%, repeat%). For each positive, a candidate from the same bin
    is selected without replacement. Falls back to nearest-neighbor for
    positives whose bin has no remaining candidates.

    Parameters
    ----------
    positive_seqs : pd.Series
        DNA sequences for positive (enhancer) regions, soft-masked.
    candidate_seqs : pd.Series
        DNA sequences for candidate negative regions, soft-masked.
        Should be oversampled (e.g. 20x) relative to positives.
    gc_bin_size : float
        Width of GC content bins (fraction, 0-1).
    repeat_bin_size : float
        Width of repeat fraction bins (fraction, 0-1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Indices into candidate_seqs, one per positive.
    """
    rng = np.random.default_rng(seed)

    pos_gc = compute_gc_content(positive_seqs).to_numpy()
    pos_repeat = compute_repeat_fraction(positive_seqs).to_numpy()
    cand_gc = compute_gc_content(candidate_seqs).to_numpy()
    cand_repeat = compute_repeat_fraction(candidate_seqs).to_numpy()

    # Assign bins
    pos_gc_bin = (pos_gc / gc_bin_size).astype(int)
    pos_rep_bin = (pos_repeat / repeat_bin_size).astype(int)
    cand_gc_bin = (cand_gc / gc_bin_size).astype(int)
    cand_rep_bin = (cand_repeat / repeat_bin_size).astype(int)

    # Build bin → candidate index mapping
    bin_to_candidates: dict[tuple[int, int], list[int]] = {}
    for i in range(len(candidate_seqs)):
        key = (cand_gc_bin[i], cand_rep_bin[i])
        bin_to_candidates.setdefault(key, []).append(i)

    # Shuffle candidates within each bin for random selection
    for indices in bin_to_candidates.values():
        rng.shuffle(indices)

    # Track consumption position per bin
    bin_pos: dict[tuple[int, int], int] = {k: 0 for k in bin_to_candidates}

    matched_indices = np.full(len(positive_seqs), -1, dtype=int)
    unmatched_positives: list[int] = []

    # Phase 1: bin-based matching
    for i in range(len(positive_seqs)):
        key = (pos_gc_bin[i], pos_rep_bin[i])
        candidates = bin_to_candidates.get(key)
        if candidates is not None and bin_pos[key] < len(candidates):
            matched_indices[i] = candidates[bin_pos[key]]
            bin_pos[key] += 1
        else:
            unmatched_positives.append(i)

    # Phase 2: nearest-neighbor fallback for unmatched positives
    if unmatched_positives:
        used = set(matched_indices[matched_indices >= 0].tolist())
        available_mask = np.ones(len(candidate_seqs), dtype=bool)
        available_mask[list(used)] = False
        available_indices = np.where(available_mask)[0]

        if len(available_indices) < len(unmatched_positives):
            msg = (
                f"Not enough candidates ({len(candidate_seqs)}) to match "
                f"all positives ({len(positive_seqs)})"
            )
            raise ValueError(msg)

        cand_props = np.column_stack(
            [cand_gc[available_indices], cand_repeat[available_indices]]
        )
        tree = KDTree(cand_props)

        # Query enough neighbors to guarantee unique assignments
        k = min(len(available_indices), len(unmatched_positives))
        queries = np.column_stack(
            [pos_gc[unmatched_positives], pos_repeat[unmatched_positives]]
        )
        _, all_nn_idx = tree.query(queries, k=k)

        # Greedily assign nearest available neighbor
        used_available: set[int] = set()
        for row, i in enumerate(unmatched_positives):
            nn_indices = all_nn_idx[row] if k > 1 else [all_nn_idx[row]]
            for nn_idx in nn_indices:
                if nn_idx not in used_available:
                    matched_indices[i] = available_indices[nn_idx]
                    used_available.add(nn_idx)
                    break

    return matched_indices

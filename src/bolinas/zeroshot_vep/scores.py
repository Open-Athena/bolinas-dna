"""Zero-shot VEP scoring functions.

Pure-numpy scoring functions called by :mod:`features` on each batch's
in-memory forward-pass outputs. Outputs are score columns suitable for
:func:`bolinas.evals.metrics.pairwise_accuracy`.

Score sign convention follows the existing leaderboards (higher = more
impactful / pathogenic) **where the natural sign is unambiguous**:

- ``llr``  = ``log p[alt] - log p[ref]`` — natural sign (positive = alt favored,
  not typically pathogenic). ``minus_llr`` / ``abs_llr`` are the actionable
  forms for the matched-pair eval; ``llr`` is included so we can sanity-check.
- ``minus_logp_ref`` / ``minus_logp_alt`` — surprisal forms (negated).
- ``entropy`` — natural sign (high entropy = ambiguous position). For mendelian
  positives we'd typically expect *low* entropy, so pairwise_accuracy(entropy)
  may end up below 0.5 — that's informative, not a bug. Flip mentally via
  ``1 - acc`` if needed (i.e. the same metric with sign reversed).
- Embedding distances — all 3 distance variants (l2, cosine, dot) use "higher =
  more different" sign. For cosine that's ``1 - cos_sim``; for dot that's
  ``-<ref, alt>``.

All score arrays come back as fp32 numpy arrays of shape ``(N,)`` and align
row-wise with the input cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Order is meaningful — `ref_idx` / `alt_idx` in the cache index into this tuple
# and into the size-4 axis of `seq_logprob`.
NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")
NUC_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(NUCLEOTIDES)}


# ---------------------------------------------------------------------------
# Likelihood scores (from joint sequence log-probs under the 4 candidate
# center nucleotides — the "bidirectional conditional" P(center | left, right)).
# ---------------------------------------------------------------------------


def _bidir_log_posterior(seq_logprob: np.ndarray) -> np.ndarray:
    """Softmax-normalize joint sequence log-probs across the 4 candidates.

    Returns ``log P(center=nuc | left, right)`` for each variant and each
    nucleotide. Shape ``(N, 4)``.
    """
    return seq_logprob - np.logaddexp.reduce(seq_logprob, axis=-1, keepdims=True)


def _entropy_nats(logp_normalized: np.ndarray) -> np.ndarray:
    """Entropy in nats of a normalized log-probability distribution.

    ``logp_normalized`` must already softmax-sum to 1 along the last axis.
    Returns shape ``(...,)``.
    """
    p = np.exp(logp_normalized)
    # 0 * log 0 = 0; mask exact zeros to avoid -inf in log.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, logp_normalized, 0.0)
    return -(p * log_p).sum(axis=-1)


def likelihood_scores(
    seq_logprob: np.ndarray,
    ref_idx: np.ndarray,
    alt_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the 6 likelihood-based scores.

    Args:
        seq_logprob: ``(N, 4)`` fp32 — joint sequence log P under each candidate
            center nucleotide, in the order :data:`NUCLEOTIDES` (``A, C, G, T``).
        ref_idx: ``(N,)`` int — index in ``{0, 1, 2, 3}`` of the ref nucleotide.
        alt_idx: ``(N,)`` int — index in ``{0, 1, 2, 3}`` of the alt nucleotide.

    Returns:
        Dict mapping score name to ``(N,)`` fp32 array. See module docstring
        for sign conventions.
    """
    assert seq_logprob.ndim == 2 and seq_logprob.shape[1] == 4, (
        f"seq_logprob must be (N, 4), got {seq_logprob.shape}"
    )
    assert ref_idx.shape == alt_idx.shape == seq_logprob.shape[:1]
    assert ((0 <= ref_idx) & (ref_idx < 4)).all()
    assert ((0 <= alt_idx) & (alt_idx < 4)).all()
    assert (ref_idx != alt_idx).all(), "ref and alt indices must differ for every variant"

    n = seq_logprob.shape[0]
    log_p = _bidir_log_posterior(seq_logprob)  # (N, 4)
    rows = np.arange(n)
    log_p_ref = log_p[rows, ref_idx].astype(np.float32)
    log_p_alt = log_p[rows, alt_idx].astype(np.float32)

    llr = log_p_alt - log_p_ref

    return {
        "llr": llr.astype(np.float32),
        "minus_llr": (-llr).astype(np.float32),
        "abs_llr": np.abs(llr).astype(np.float32),
        "minus_logp_ref": (-log_p_ref).astype(np.float32),
        "minus_logp_alt": (-log_p_alt).astype(np.float32),
        "entropy": _entropy_nats(log_p).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Embedding distance scores (24 = 3 distances × 4 pool strategies × 2 layers).
# ---------------------------------------------------------------------------


POOLS: tuple[str, ...] = ("flat", "mean", "varpos", "lastpos")
LAYERS: tuple[str, ...] = ("last", "middle")
DISTANCES: tuple[str, ...] = ("l2", "cosine", "dot")


def _pool(emb: np.ndarray, pool: str, var_pos: int) -> np.ndarray:
    """Reduce ``(N, L, D)`` to ``(N, D_out)`` per pool strategy.

    - ``flat``: no reduction; flatten to ``(N, L*D)``.
    - ``mean``: average over positions to ``(N, D)``.
    - ``varpos``: pick position ``var_pos`` only, ``(N, D)``.
    - ``lastpos``: pick the last position ``L-1``, ``(N, D)``.
    """
    if pool == "flat":
        return emb.reshape(emb.shape[0], -1)
    if pool == "mean":
        return emb.mean(axis=1)
    if pool == "varpos":
        assert 0 <= var_pos < emb.shape[1], (
            f"var_pos={var_pos} out of range for L={emb.shape[1]}"
        )
        return emb[:, var_pos, :]
    if pool == "lastpos":
        return emb[:, -1, :]
    raise ValueError(f"unknown pool {pool!r}, expected one of {POOLS}")


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return np.sqrt((diff * diff).sum(axis=-1))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``1 - cos_sim``, row-wise. Higher = more different. Range ``[0, 2]``."""
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    num = (a32 * b32).sum(axis=-1)
    den = np.sqrt((a32 * a32).sum(axis=-1)) * np.sqrt((b32 * b32).sum(axis=-1))
    # All-zero vectors would be model degeneracy; clamp den to avoid div-by-zero
    # silently propagating NaNs into PairwiseAccuracy (which has a no-NaN check).
    den = np.maximum(den, 1e-12)
    return (1.0 - num / den).astype(np.float32)


def _neg_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``-<a, b>``, row-wise. Higher = less similar."""
    return -(a.astype(np.float32) * b.astype(np.float32)).sum(axis=-1)


_DIST_FNS = {
    "l2": _l2,
    "cosine": _cosine_dist,
    "dot": _neg_dot,
}


def embedding_scores(
    emb_ref_last: np.ndarray,
    emb_ref_middle: np.ndarray,
    emb_alt_last: np.ndarray,
    emb_alt_middle: np.ndarray,
    var_pos: int,
) -> dict[str, np.ndarray]:
    """Compute the 24 embedding distance scores.

    Args:
        emb_ref_last, emb_ref_middle, emb_alt_last, emb_alt_middle: each
            ``(N, L, D)`` fp16 or fp32 — per-position hidden states from the
            REF and ALT forward passes for last and middle layers respectively.
        var_pos: int — index of the variant center position within the
            ``L``-length embedding tensor. For a causal LM where the genome
            window is centered at the variant and the tokenizer prepends BOS,
            this is ``window_size // 2 + n_prefix``. Without BOS it's just
            ``window_size // 2``.

    Returns:
        Dict mapping score names of the form ``embed_{distance}_{pool}_{layer}``
        to ``(N,)`` fp32 arrays.
    """
    embs = {
        "last": (emb_ref_last, emb_alt_last),
        "middle": (emb_ref_middle, emb_alt_middle),
    }
    # Shape-check up front — these tensors are large enough that a downstream
    # error after pooling is annoying to debug.
    for layer_name, (ref, alt) in embs.items():
        assert ref.ndim == 3, f"{layer_name}-layer ref emb must be 3D, got {ref.shape}"
        assert alt.shape == ref.shape, (
            f"{layer_name}-layer ref/alt shape mismatch: {ref.shape} vs {alt.shape}"
        )

    results: dict[str, np.ndarray] = {}
    for layer_name, (ref, alt) in embs.items():
        for pool in POOLS:
            ref_p = _pool(ref, pool, var_pos)
            alt_p = _pool(alt, pool, var_pos)
            for dist in DISTANCES:
                results[f"embed_{dist}_{pool}_{layer_name}"] = _DIST_FNS[dist](
                    ref_p, alt_p
                ).astype(np.float32)
    return results


# ---------------------------------------------------------------------------
# Driver: cache → score DataFrame.
# ---------------------------------------------------------------------------


SCORE_NAMES: list[str] = [
    "llr",
    "minus_llr",
    "abs_llr",
    "minus_logp_ref",
    "minus_logp_alt",
    "entropy",
] + [
    f"embed_{dist}_{pool}_{layer}"
    for layer in LAYERS
    for pool in POOLS
    for dist in DISTANCES
]
assert len(SCORE_NAMES) == 30, f"expected 30 scores, got {len(SCORE_NAMES)}"


def all_scores(
    seq_logprob: np.ndarray,
    ref_idx: np.ndarray,
    alt_idx: np.ndarray,
    emb_ref_last: np.ndarray,
    emb_ref_middle: np.ndarray,
    emb_alt_last: np.ndarray,
    emb_alt_middle: np.ndarray,
    var_pos: int,
) -> pd.DataFrame:
    """Compute all 30 base score columns. Returns a ``(N, 30)`` DataFrame."""
    cols: dict[str, np.ndarray] = {}
    cols.update(likelihood_scores(seq_logprob, ref_idx, alt_idx))
    cols.update(
        embedding_scores(
            emb_ref_last,
            emb_ref_middle,
            emb_alt_last,
            emb_alt_middle,
            var_pos,
        )
    )
    df = pd.DataFrame(cols)
    # Keep the canonical order so downstream tables / heatmaps stay consistent.
    df = df[SCORE_NAMES]
    assert not df.isna().any().any(), "scores contain NaN — investigate before metrics"
    return df



"""Unit tests for bolinas.zeroshot_vep.scores.

Synthetic inputs with hand-computed expected values for each score family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bolinas.zeroshot_vep import scores


# ---------------------------------------------------------------------------
# Likelihood scores
# ---------------------------------------------------------------------------


def test_llr_matches_difference_of_seq_logprob():
    # Construct seq_logprob with known values; LLR should be alt - ref.
    seq_logprob = np.array(
        [
            [-1.0, -2.0, -3.0, -4.0],  # ref=A (0), alt=C (1)  → llr = -2.0 - (-1.0) = -1.0
            [-5.0, -10.0, -1.0, -2.0],  # ref=G (2), alt=A (0)  → llr = -5.0 - (-1.0) = -4.0
        ],
        dtype=np.float32,
    )
    ref_idx = np.array([0, 2], dtype=np.int64)
    alt_idx = np.array([1, 0], dtype=np.int64)
    s = scores.likelihood_scores(seq_logprob, ref_idx, alt_idx)
    assert np.allclose(s["llr"], [-1.0, -4.0])
    assert np.allclose(s["minus_llr"], [1.0, 4.0])
    assert np.allclose(s["abs_llr"], [1.0, 4.0])


def test_llr_invariant_to_seq_logprob_offset():
    # Adding a constant to all candidate seq_logprobs (which would happen e.g.
    # under a different prefix log-prob normalization) must not change LLR.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((50, 4)).astype(np.float32)
    offset = rng.standard_normal(50).astype(np.float32)[:, None]
    ref_idx = rng.integers(0, 4, size=50)
    alt_idx = (ref_idx + 1 + rng.integers(0, 3, size=50)) % 4  # always differs
    s_base = scores.likelihood_scores(base, ref_idx, alt_idx)
    s_shifted = scores.likelihood_scores(base + offset, ref_idx, alt_idx)
    np.testing.assert_allclose(s_base["llr"], s_shifted["llr"], atol=1e-5)


def test_entropy_uniform_is_log_4():
    # Uniform conditional → entropy = log 4 ≈ 1.386 nats.
    seq_logprob = np.zeros((3, 4), dtype=np.float32)  # any constant works (all equal)
    ref_idx = np.array([0, 1, 2], dtype=np.int64)
    alt_idx = np.array([1, 2, 3], dtype=np.int64)
    s = scores.likelihood_scores(seq_logprob, ref_idx, alt_idx)
    np.testing.assert_allclose(s["entropy"], np.log(4), atol=1e-6)


def test_entropy_peaked_is_zero():
    # One candidate strongly dominates → entropy → 0.
    seq_logprob = np.array([[0.0, -100.0, -100.0, -100.0]], dtype=np.float32)
    ref_idx = np.array([0], dtype=np.int64)
    alt_idx = np.array([1], dtype=np.int64)
    s = scores.likelihood_scores(seq_logprob, ref_idx, alt_idx)
    assert s["entropy"][0] < 1e-30


def test_minus_logp_ref_and_alt_match_normalized_posterior():
    # minus_logp_ref / minus_logp_alt are -log of the SOFTMAX-NORMALIZED posterior,
    # not -log of the raw joint seq_logprob.
    seq_logprob = np.array([[0.0, np.log(3.0), 0.0, 0.0]], dtype=np.float32)
    # softmax = [1, 3, 1, 1] / 6 = [1/6, 3/6, 1/6, 1/6]; log → [-log 6, log 3 - log 6, ...]
    ref_idx = np.array([0], dtype=np.int64)  # ref=A → p=1/6
    alt_idx = np.array([1], dtype=np.int64)  # alt=C → p=3/6
    s = scores.likelihood_scores(seq_logprob, ref_idx, alt_idx)
    np.testing.assert_allclose(s["minus_logp_ref"], [np.log(6)], atol=1e-6)
    np.testing.assert_allclose(s["minus_logp_alt"], [np.log(2)], atol=1e-6)  # = log(6/3)


def test_likelihood_rejects_equal_ref_alt():
    seq_logprob = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(AssertionError, match="ref and alt indices must differ"):
        scores.likelihood_scores(
            seq_logprob,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Embedding scores
# ---------------------------------------------------------------------------


def _toy_embeddings(n=2, L=5, D=3):
    """Generate ref/alt embeddings with known structure for assertions."""
    rng = np.random.default_rng(42)
    ref_last = rng.standard_normal((n, L, D)).astype(np.float32)
    ref_middle = rng.standard_normal((n, L, D)).astype(np.float32)
    # alt = ref + epsilon at variant position → only varpos pool sees difference,
    # flat / mean / lastpos see attenuated differences.
    alt_last = ref_last.copy()
    alt_middle = ref_middle.copy()
    return ref_last, ref_middle, alt_last, alt_middle


def test_embed_l2_zero_when_ref_equals_alt():
    ref_last, ref_middle, alt_last, alt_middle = _toy_embeddings()
    s = scores.embedding_scores(ref_last, ref_middle, alt_last, alt_middle, var_pos=2)
    for name, vals in s.items():
        if name.startswith("embed_l2_"):
            np.testing.assert_allclose(vals, 0.0, atol=1e-6, err_msg=name)


def test_embed_cosine_zero_when_ref_equals_alt():
    ref_last, ref_middle, alt_last, alt_middle = _toy_embeddings()
    s = scores.embedding_scores(ref_last, ref_middle, alt_last, alt_middle, var_pos=2)
    for name, vals in s.items():
        if name.startswith("embed_cosine_"):
            np.testing.assert_allclose(vals, 0.0, atol=1e-5, err_msg=name)


def test_embed_l2_varpos_only_sees_variant_position():
    n, L, D = 2, 5, 3
    ref_last = np.zeros((n, L, D), dtype=np.float32)
    ref_middle = np.zeros((n, L, D), dtype=np.float32)
    alt_last = np.zeros((n, L, D), dtype=np.float32)
    alt_middle = np.zeros((n, L, D), dtype=np.float32)
    # Perturb alt by [3, 4, 0] at position 2 → L2 distance at varpos = 5.
    alt_last[:, 2, 0] = 3.0
    alt_last[:, 2, 1] = 4.0
    s = scores.embedding_scores(ref_last, ref_middle, alt_last, alt_middle, var_pos=2)
    np.testing.assert_allclose(s["embed_l2_varpos_last"], [5.0, 5.0], atol=1e-5)
    # flat distance should equal varpos here since the only nonzero is at varpos.
    np.testing.assert_allclose(s["embed_l2_flat_last"], [5.0, 5.0], atol=1e-5)
    # mean-pool: difference is [3/L, 4/L, 0] → L2 = 5/L.
    np.testing.assert_allclose(s["embed_l2_mean_last"], [5.0 / L, 5.0 / L], atol=1e-5)
    # lastpos: position L-1 = 4, where there's no perturbation → distance 0.
    np.testing.assert_allclose(s["embed_l2_lastpos_last"], [0.0, 0.0], atol=1e-5)
    # middle layer untouched → all zero.
    np.testing.assert_allclose(s["embed_l2_varpos_middle"], [0.0, 0.0], atol=1e-5)


def test_embed_cosine_antiparallel_is_2():
    # Antiparallel unit vectors → cos_sim = -1 → distance = 2.
    n, L, D = 1, 1, 2
    ref = np.zeros((n, L, D), dtype=np.float32)
    alt = np.zeros((n, L, D), dtype=np.float32)
    ref[0, 0, :] = [1.0, 0.0]
    alt[0, 0, :] = [-1.0, 0.0]
    s = scores.embedding_scores(ref, ref, alt, alt, var_pos=0)
    np.testing.assert_allclose(s["embed_cosine_varpos_last"], [2.0], atol=1e-5)
    np.testing.assert_allclose(s["embed_cosine_varpos_middle"], [2.0], atol=1e-5)


def test_embed_dot_higher_when_more_different():
    # Two parallel vs antiparallel cases → neg-dot should be more positive for antiparallel.
    n, L, D = 2, 1, 2
    ref = np.zeros((n, L, D), dtype=np.float32)
    alt = np.zeros((n, L, D), dtype=np.float32)
    ref[0, 0, :] = [1.0, 0.0]
    alt[0, 0, :] = [1.0, 0.0]  # identical → dot = 1, neg_dot = -1
    ref[1, 0, :] = [1.0, 0.0]
    alt[1, 0, :] = [-1.0, 0.0]  # antiparallel → dot = -1, neg_dot = 1
    s = scores.embedding_scores(ref, ref, alt, alt, var_pos=0)
    np.testing.assert_allclose(s["embed_dot_varpos_last"], [-1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# all_scores driver
# ---------------------------------------------------------------------------


def test_all_scores_returns_30_columns():
    n, L, D = 4, 8, 3
    rng = np.random.default_rng(0)
    seq_logprob = rng.standard_normal((n, 4)).astype(np.float32)
    ref_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    alt_idx = np.array([1, 2, 3, 0], dtype=np.int64)
    ref_last = rng.standard_normal((n, L, D)).astype(np.float32)
    ref_middle = rng.standard_normal((n, L, D)).astype(np.float32)
    alt_last = rng.standard_normal((n, L, D)).astype(np.float32)
    alt_middle = rng.standard_normal((n, L, D)).astype(np.float32)
    df = scores.all_scores(
        seq_logprob, ref_idx, alt_idx,
        ref_last, ref_middle, alt_last, alt_middle,
        var_pos=L // 2,
    )
    assert df.shape == (n, 30)
    assert list(df.columns) == scores.SCORE_NAMES
    assert not df.isna().any().any()


def test_all_scores_column_naming_grid():
    # 24 embed scores = 3 distances × 4 pools × 2 layers — exhaustively present.
    expected = {
        f"embed_{dist}_{pool}_{layer}"
        for dist in scores.DISTANCES
        for pool in scores.POOLS
        for layer in scores.LAYERS
    }
    assert expected.issubset(set(scores.SCORE_NAMES))
    assert len(expected) == 24


def test_score_directions_renames_match_expected():
    # The three locked renames: entropy → minus_entropy, embed_dot_* → embed_minus_dot_*,
    # minus_logp_ref → logp_ref. Everything else passes through unchanged.
    assert "minus_entropy" in scores.SCORE_DIRECTIONS
    assert scores.SCORE_DIRECTIONS["minus_entropy"] == ("entropy", -1.0)
    assert "logp_ref" in scores.SCORE_DIRECTIONS
    assert scores.SCORE_DIRECTIONS["logp_ref"] == ("minus_logp_ref", -1.0)
    # All 8 dot scores renamed to minus_dot, with sign +1 (already negated upstream).
    for pool in scores.POOLS:
        for layer in scores.LAYERS:
            final = f"embed_minus_dot_{pool}_{layer}"
            raw = f"embed_dot_{pool}_{layer}"
            assert scores.SCORE_DIRECTIONS[final] == (raw, 1.0), final
            assert f"embed_dot_{pool}_{layer}" not in scores.SCORE_DIRECTIONS, (
                "raw embed_dot_* should not appear as a final-name key"
            )
    # Total final names = 30.
    assert len(scores.SCORE_DIRECTIONS) == 30
    assert len(scores.SCORE_NAMES_FINAL) == 30


def test_apply_score_directions_flips_correctly():
    rng = np.random.default_rng(0)
    n, L, D = 5, 8, 3
    seq_logprob = rng.standard_normal((n, 4)).astype(np.float32)
    ref_idx = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    alt_idx = np.array([1, 2, 3, 0, 2], dtype=np.int64)
    emb_args = [rng.standard_normal((n, L, D)).astype(np.float32) for _ in range(4)]
    raw_df = scores.all_scores(seq_logprob, ref_idx, alt_idx, *emb_args, var_pos=L // 2)

    signed = scores.apply_score_directions(raw_df)
    # entropy → minus_entropy with sign flip
    np.testing.assert_allclose(signed["minus_entropy"], -raw_df["entropy"], rtol=1e-5)
    # minus_logp_ref → logp_ref with sign flip
    np.testing.assert_allclose(signed["logp_ref"], -raw_df["minus_logp_ref"], rtol=1e-5)
    # embed_dot_* → embed_minus_dot_*: rename only, no flip
    for pool in scores.POOLS:
        for layer in scores.LAYERS:
            np.testing.assert_allclose(
                signed[f"embed_minus_dot_{pool}_{layer}"],
                raw_df[f"embed_dot_{pool}_{layer}"],
                rtol=1e-5,
            )
    # Pass-through columns
    np.testing.assert_allclose(signed["minus_llr"], raw_df["minus_llr"])
    np.testing.assert_allclose(signed["abs_llr"], raw_df["abs_llr"])
    np.testing.assert_allclose(signed["embed_l2_flat_last"], raw_df["embed_l2_flat_last"])


def test_apply_score_directions_missing_raw_column_raises():
    bad = pd.DataFrame({"llr": [0.1, 0.2]})
    with pytest.raises(KeyError, match="raw score column"):
        scores.apply_score_directions(bad)

"""Tests for the AlphaGenome scoring helpers (issue #154).

The ``alphagenome`` PyPI package is an optional dep (``alphagenome-eval`` group)
and may not be installed locally. We isolate tests so the parser test runs
without it; the scorer-construction test is gated on the import.
"""

import pandas as pd
import pytest

from bolinas.evals.alphagenome import (
    ALPHAGENOME_TRACKS,
    SEQUENCE_LENGTH,
    parse_score_response,
)


def test_alphagenome_constants():
    assert ALPHAGENOME_TRACKS == (
        "ATAC",
        "DNASE",
        "CHIP_TF",
        "CHIP_HISTONE",
        "CAGE",
        "PROCAP",
        "RNA_SEQ",
    )
    assert SEQUENCE_LENGTH == "1MB"


def test_parse_score_response_single_track_per_assay():
    """Each assay returns one cell-type/track row → columns named ASSAY_0."""
    scorer_repr_to_assay = {
        "scorer_atac": "ATAC",
        "scorer_cage": "CAGE",
    }
    tidy = pd.DataFrame(
        {
            "variant_scorer": ["scorer_atac", "scorer_cage"],
            "raw_score": [1.5, 2.5],
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert out.shape == (1, 2)
    assert set(out.columns) == {"ATAC_0", "CAGE_0"}
    assert out.loc[0, "ATAC_0"] == 1.5
    assert out.loc[0, "CAGE_0"] == 2.5


def test_parse_score_response_multiple_tracks_per_assay():
    """An assay with multiple cell-type tracks → ASSAY_0, ASSAY_1, ..."""
    scorer_repr_to_assay = {"scorer_atac": "ATAC", "scorer_cage": "CAGE"}
    tidy = pd.DataFrame(
        {
            # ATAC has 3 cell-type tracks; CAGE has 2.
            "variant_scorer": [
                "scorer_atac",
                "scorer_atac",
                "scorer_atac",
                "scorer_cage",
                "scorer_cage",
            ],
            "raw_score": [0.1, 0.2, 0.3, 1.0, 2.0],
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert out.shape == (1, 5)
    assert list(out.columns) == ["ATAC_0", "ATAC_1", "ATAC_2", "CAGE_0", "CAGE_1"]
    # Order within each assay reflects the row order in tidy_scores.
    assert out.loc[0, "ATAC_0"] == 0.1
    assert out.loc[0, "ATAC_2"] == 0.3
    assert out.loc[0, "CAGE_1"] == 2.0


def test_parse_score_response_unknown_scorer_fails_loud():
    """A scorer repr we don't know about should crash, not silently drop."""
    scorer_repr_to_assay = {"scorer_atac": "ATAC"}
    tidy = pd.DataFrame(
        {
            "variant_scorer": ["scorer_atac", "scorer_unknown"],
            "raw_score": [1.0, 2.0],
        }
    )
    with pytest.raises(AssertionError, match="not in scorer_repr_to_assay"):
        parse_score_response(tidy, scorer_repr_to_assay)


def test_parse_score_response_missing_columns():
    scorer_repr_to_assay = {"scorer_atac": "ATAC"}
    tidy = pd.DataFrame({"variant_scorer": ["scorer_atac"], "score": [1.0]})
    with pytest.raises(AssertionError, match="unexpected tidy_scores columns"):
        parse_score_response(tidy, scorer_repr_to_assay)


def test_parse_score_response_no_nan_in_normal_path():
    """Normal AlphaGenome output should never produce NaN columns."""
    scorer_repr_to_assay = {f"s_{t}": t for t in ALPHAGENOME_TRACKS}
    tidy = pd.DataFrame(
        {
            "variant_scorer": [f"s_{t}" for t in ALPHAGENOME_TRACKS],
            "raw_score": list(range(len(ALPHAGENOME_TRACKS))),
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert not out.isna().any().any()
    assert len(out.columns) == len(ALPHAGENOME_TRACKS)


def test_make_scorers_uses_l2_diff_log1p():
    """If alphagenome is installed, scorers must request L2_DIFF_LOG1P aggregation."""
    pytest.importorskip("alphagenome")
    from alphagenome.models import variant_scorers

    from bolinas.evals.alphagenome import make_scorers

    scorers, repr_to_assay = make_scorers()
    assert len(scorers) == len(ALPHAGENOME_TRACKS)
    assert set(repr_to_assay.values()) == set(ALPHAGENOME_TRACKS)
    for s in scorers:
        # CenterMaskScorer exposes its config; verify aggregation is L2_DIFF_LOG1P.
        assert s.aggregation_type == variant_scorers.AggregationType.L2_DIFF_LOG1P
        # width=None means use the scorer's default span.
        assert s.width is None

"""Tests for Evo2 variant scoring helpers.

The real inference path needs ``evo2`` + an H100 + ~40 GB of downloaded
weights, so this file only exercises the lightweight contract (signature,
import, score-dataframe shape). The heavy lifting is checked at runtime on
the SkyPilot cluster.
"""

import numpy as np
import pandas as pd

from bolinas.evals.evo2 import compute_evo2_llr, scores_dataframe


def test_compute_evo2_llr_signature_and_imports():
    """Sanity-check that the function exists and is importable without evo2 installed."""
    assert callable(compute_evo2_llr)


def test_scores_dataframe_shapes_and_signs():
    llr = np.array([-1.5, 0.0, 2.0])
    scores = scores_dataframe(llr)

    assert list(scores.columns) == ["llr", "minus_llr", "abs_llr"]
    assert len(scores) == 3
    pd.testing.assert_series_equal(
        scores["minus_llr"], pd.Series([1.5, -0.0, -2.0], name="minus_llr")
    )
    pd.testing.assert_series_equal(
        scores["abs_llr"], pd.Series([1.5, 0.0, 2.0], name="abs_llr")
    )

"""Threshold calibration: match the genome-wide passing-nucleotide count of
one phyloP track to that of another at a fixed reference threshold.

Use case: pick a threshold for ``phyloP_447m`` such that the number of bases
genome-wide with ``phyloP_447m >= T`` equals the number of bases with
``phyloP_241m >= 2.27`` (the canonical 241m threshold used elsewhere in the
repo).
"""

from __future__ import annotations

from .histogram import PhylopHistogram


def calibrate_to_match_count(
    target_hist: PhylopHistogram,
    ref_hist: PhylopHistogram,
    ref_threshold: float,
    *,
    target_name: str = "target",
    ref_name: str = "reference",
) -> dict:
    """Find ``T`` for ``target_hist`` such that ``target_hist.count_ge(T) ≈ ref_count``.

    ``ref_count = ref_hist.count_ge(ref_threshold)``. The returned threshold
    is linearly interpolated within the bracketing bin of ``target_hist``;
    precision is bounded by the bin width.

    Returns a JSON-serialisable dict with both tracks' thresholds, counts,
    the relative error, and the bin metadata. Asserts that the absolute
    relative error between the matched count and the reference count is
    under 1% — well below the bin-width-driven precision floor for sensible
    bin counts.
    """
    ref_count = ref_hist.count_ge(ref_threshold)
    target_threshold = target_hist.threshold_for_count(ref_count)
    target_count = target_hist.count_ge(target_threshold)

    if ref_count > 0:
        rel_err = abs(target_count / ref_count - 1.0)
    else:
        rel_err = 0.0
    assert rel_err < 0.01, (
        f"calibration failed: {target_name} count {target_count} vs "
        f"{ref_name} count {ref_count} (rel err {rel_err:.4f}); "
        f"likely the histogram bin width is too coarse."
    )

    return {
        ref_name: {
            "threshold": float(ref_threshold),
            "count": int(ref_count),
            "total_bases": int(ref_hist.total()),
            "n_nan": int(ref_hist.n_nan),
        },
        target_name: {
            "threshold": float(target_threshold),
            "count": int(target_count),
            "count_target": int(ref_count),
            "abs_relative_error": float(rel_err),
            "total_bases": int(target_hist.total()),
            "n_nan": int(target_hist.n_nan),
        },
        "hist_meta": {
            "n_bins": int(target_hist.n_bins),
            "min": float(target_hist.edges[0]),
            "max": float(target_hist.edges[-1]),
            "bin_width": float(target_hist.edges[1] - target_hist.edges[0]),
        },
    }

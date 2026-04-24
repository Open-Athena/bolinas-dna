"""Tests for top-k misclassified-bin ranking and overlap annotation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyBigWig
import pytest

from bolinas.enhancer_segmentation.misclassified import top_misclassified_bins

BIN_SIZE = 128


def _bins_for_chrom(chrom: str, genome: str, n: int) -> dict:
    """Build a long-form prediction frame for ``n`` bins on ``chrom`` with
    mixed label / logit patterns useful for ranking tests."""
    starts = np.arange(n, dtype=np.int64) * BIN_SIZE
    ends = starts + BIN_SIZE
    return {
        "genome": [genome] * n,
        "chrom": [chrom] * n,
        "bin_start": starts.tolist(),
        "bin_end": ends.tolist(),
    }


def test_fp_and_fn_ranking_picks_worst_bins():
    """FP = label=0 sorted desc by logit; FN = label=1 sorted asc by logit.
    Ties broken by original order.
    """
    n = 20
    base = _bins_for_chrom("1", "homo_sapiens", n)
    # Alternate labels; logits increase with index so highest-logit negatives
    # are at the tail, lowest-logit positives at the head.
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8)
    logits = np.linspace(-5.0, 5.0, n, dtype=np.float32)
    preds = pl.DataFrame(
        {**base, "label": labels, "logit": logits}
    ).cast({"label": pl.UInt8, "logit": pl.Float32})

    out = top_misclassified_bins(
        preds,
        exons_by_species={"homo_sapiens": pl.DataFrame(schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64})},
        all_cres_by_species={"homo_sapiens": pl.DataFrame(schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "cre_class": pl.Utf8})},
        conservation_bw_by_species={},  # skip bigwig to keep test hermetic
        top_k=3,
    )
    fp = out.filter(pl.col("error_type") == "false_positive").sort("logit", descending=True)
    fn = out.filter(pl.col("error_type") == "false_negative").sort("logit")
    # Top-3 FP are the 3 highest-logit label=0 bins. labels are 0 at even
    # indices, and logits increase with index, so top-3 FP are indices 18, 16, 14.
    assert fp["bin_start"].to_list() == [18 * BIN_SIZE, 16 * BIN_SIZE, 14 * BIN_SIZE]
    # Top-3 FN: lowest-logit label=1 bins at odd indices 1, 3, 5.
    assert fn["bin_start"].to_list() == [1 * BIN_SIZE, 3 * BIN_SIZE, 5 * BIN_SIZE]
    assert fp["label"].to_list() == [0, 0, 0]
    assert fn["label"].to_list() == [1, 1, 1]


def test_probability_computed_from_logit():
    preds = pl.DataFrame(
        {
            "genome": ["h"],
            "chrom": ["1"],
            "bin_start": [0],
            "bin_end": [BIN_SIZE],
            "label": [0],
            "logit": [0.0],
        }
    ).cast({"label": pl.UInt8, "logit": pl.Float32})
    out = top_misclassified_bins(
        preds,
        exons_by_species={},
        all_cres_by_species={},
        conservation_bw_by_species={},
        top_k=5,
    )
    assert out["probability"].to_list() == [pytest.approx(0.5)]


def test_exon_and_cre_annotations_detect_overlaps():
    # Three bins on chr1: bin 0 = [0,128), bin 1 = [128,256), bin 2 = [256,384).
    preds = pl.DataFrame(
        {
            "genome": ["h", "h", "h"],
            "chrom": ["1", "1", "1"],
            "bin_start": [0, 128, 256],
            "bin_end": [128, 256, 384],
            "label": [0, 0, 0],
            "logit": [3.0, 2.0, 1.0],  # ranks: bin0, bin1, bin2
        }
    ).cast({"label": pl.UInt8, "logit": pl.Float32})
    exons = pl.DataFrame(
        {"chrom": ["1"], "start": [64], "end": [192]}  # touches bins 0, 1
    )
    cres = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [300, 100],
            "end": [350, 125],  # CA-CTCF stays inside bin 0 only
            "cre_class": ["dELS", "CA-CTCF"],
        }
    )
    out = top_misclassified_bins(
        preds,
        exons_by_species={"h": exons},
        all_cres_by_species={"h": cres},
        conservation_bw_by_species={},
        top_k=3,
    ).filter(pl.col("error_type") == "false_positive").sort("logit", descending=True)
    assert out["overlaps_exon"].to_list() == [True, True, False]
    assert out["overlaps_any_cre"].to_list() == [True, False, True]
    # bin 0 gets CA-CTCF; bin 2 gets dELS; bin 1 gets nothing.
    assert out["overlapping_cre_classes"].to_list() == ["CA-CTCF", None, "dELS"]


def test_multi_class_overlap_joined_sorted_and_deduped():
    preds = pl.DataFrame(
        {
            "genome": ["h"],
            "chrom": ["1"],
            "bin_start": [0],
            "bin_end": [128],
            "label": [0],
            "logit": [5.0],
        }
    ).cast({"label": pl.UInt8, "logit": pl.Float32})
    cres = pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "start": [0, 10, 50],
            "end": [20, 40, 100],
            "cre_class": ["pELS", "dELS", "pELS"],  # duplicate pELS
        }
    )
    out = top_misclassified_bins(
        preds,
        exons_by_species={},
        all_cres_by_species={"h": cres},
        conservation_bw_by_species={},
        top_k=1,
    )
    assert out["overlapping_cre_classes"].to_list() == ["dELS,pELS"]


def test_per_species_topk_is_independent(tmp_path):
    # Two species, each has its own top-k. Even if one species has much more
    # extreme logits, the other species should still get its own top-k.
    n = 10
    rows = []
    for genome, logit_scale in [("h", 10.0), ("m", 1.0)]:
        for i in range(n):
            rows.append(
                {
                    "genome": genome,
                    "chrom": "1",
                    "bin_start": i * BIN_SIZE,
                    "bin_end": (i + 1) * BIN_SIZE,
                    "label": i % 2,
                    "logit": float(i - n / 2) * logit_scale,
                }
            )
    preds = pl.DataFrame(rows).cast({"label": pl.UInt8, "logit": pl.Float32})
    out = top_misclassified_bins(
        preds,
        exons_by_species={},
        all_cres_by_species={},
        conservation_bw_by_species={},
        top_k=2,
    )
    per_species_fp = out.filter(pl.col("error_type") == "false_positive").group_by("genome").len()
    assert sorted(per_species_fp["genome"].to_list()) == ["h", "m"]
    assert per_species_fp["len"].to_list() == [2, 2]


def test_mean_phastcons_from_bigwig(tmp_path):
    bw_path = str(tmp_path / "cons.bw")
    bw = pyBigWig.open(bw_path, "w")
    bw.addHeader([("chr1", 1000)])
    # bin 0 = [0, 128): score = 1.0 everywhere
    # bin 1 = [128, 256): no coverage -> NaN, treated as 0
    bw.addEntries(["chr1"], [0], ends=[128], values=[1.0])
    bw.close()
    preds = pl.DataFrame(
        {
            "genome": ["h", "h"],
            "chrom": ["1", "1"],
            "bin_start": [0, 128],
            "bin_end": [128, 256],
            "label": [0, 0],
            "logit": [2.0, 1.0],
        }
    ).cast({"label": pl.UInt8, "logit": pl.Float32})
    out = top_misclassified_bins(
        preds,
        exons_by_species={},
        all_cres_by_species={},
        conservation_bw_by_species={"h": bw_path},
        top_k=2,
    ).sort("logit", descending=True)
    assert out["mean_phastcons"].to_list() == [pytest.approx(1.0), pytest.approx(0.0)]


def test_handles_val_predictions_with_window_and_bin_columns():
    """Regression: the real val_predictions parquet has both window-level
    (``start``, ``end``) *and* bin-level (``bin_start``, ``bin_end``) columns.
    The overlap helpers must build their bedframe from the bin columns only,
    without renaming into a collision with the window columns.
    """
    preds = pl.DataFrame(
        {
            "genome": ["h"] * 3,
            "chrom": ["1"] * 3,
            # Window-level coords (length = 65536).
            "start": [0, 0, 0],
            "end": [65536, 65536, 65536],
            "strand": ["+"] * 3,
            "bin_idx": [0, 1, 2],
            "bin_start": [0, 128, 256],
            "bin_end": [128, 256, 384],
            "label": [0, 0, 1],
            "logit": [5.0, 2.0, -3.0],
        }
    ).cast({"label": pl.UInt8, "logit": pl.Float32, "bin_idx": pl.Int32})
    exons = pl.DataFrame({"chrom": ["1"], "start": [100], "end": [200]})
    cres = pl.DataFrame(
        {"chrom": ["1"], "start": [250], "end": [300], "cre_class": ["dELS"]}
    )
    out = top_misclassified_bins(
        preds,
        exons_by_species={"h": exons},
        all_cres_by_species={"h": cres},
        conservation_bw_by_species={},
        top_k=3,
    )
    # Don't care about exact ordering; just verify the call doesn't crash and
    # annotations are plausible.
    assert out.height == 3
    fp = out.filter(pl.col("error_type") == "false_positive").sort("logit", descending=True)
    # Highest-logit FP bin is at bin_start=0 (logit=5.0): overlaps exon
    # [100,200) because exon crosses bin [0,128); does NOT overlap dELS.
    row0 = fp.row(0, named=True)
    assert row0["bin_start"] == 0
    assert row0["overlaps_exon"] is True
    assert row0["overlaps_any_cre"] is False


def test_empty_predictions_returns_empty_frame():
    preds = pl.DataFrame(
        schema={
            "genome": pl.Utf8,
            "chrom": pl.Utf8,
            "bin_start": pl.Int64,
            "bin_end": pl.Int64,
            "label": pl.UInt8,
            "logit": pl.Float32,
        }
    )
    out = top_misclassified_bins(
        preds,
        exons_by_species={},
        all_cres_by_species={},
        conservation_bw_by_species={},
        top_k=10,
    )
    assert out.height == 0
    assert "error_type" in out.columns

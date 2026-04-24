"""Top-k misclassified bins for the per-bin segmentation model.

For a trained segmenter's full-coverage validation predictions, rank bins by
their error signal:

- **False positives**: bins with ``label == 0`` sorted by descending logit
  (most confidently predicted enhancer, actually negative).
- **False negatives**: bins with ``label == 1`` sorted by ascending logit
  (least confidently predicted enhancer, actually positive).

Each top-k bin is annotated with overlap flags against exons and the full
SCREEN cCRE registry, plus a mean phastCons score, so the user can eyeball
*why* the model is confused there (e.g. bin straddles an exon; or a CA-CTCF
cCRE the model is confidently picking up even though only ELS is positive).
"""

from __future__ import annotations

import bioframe as bf
import numpy as np
import polars as pl
import pyBigWig


def _prob_from_logit(logit: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid. Avoid scipy.special.expit on float32 so test
    # fixtures don't need the extra dep surface.
    return 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))


def _overlap_flags(
    bins: pl.DataFrame, intervals: pl.DataFrame, chrom_col: str = "chrom"
) -> np.ndarray:
    """Return a boolean numpy array of length ``len(bins)`` — True iff the
    bin overlaps any row of ``intervals`` (same-chromosome, half-open).

    ``bins`` must have columns (``chrom``, ``bin_start``, ``bin_end``).
    ``intervals`` must have columns (``chrom``, ``start``, ``end``).
    """
    if intervals.height == 0 or bins.height == 0:
        return np.zeros(bins.height, dtype=bool)
    b = bins.to_pandas().rename(columns={"bin_start": "start", "bin_end": "end"})
    i = intervals.to_pandas()[[chrom_col, "start", "end"]]
    b[chrom_col] = b[chrom_col].astype(str)
    i[chrom_col] = i[chrom_col].astype(str)
    counts = bf.count_overlaps(b, i)["count"].to_numpy()
    return counts > 0


def _overlap_cre_classes(
    bins: pl.DataFrame, all_cres: pl.DataFrame
) -> list[str | None]:
    """For each bin, return a comma-separated sorted string of the SCREEN
    cre_class values it overlaps (or None if none). Assumes ``all_cres`` has
    columns (``chrom``, ``start``, ``end``, ``cre_class``).
    """
    if all_cres.height == 0 or bins.height == 0:
        return [None] * bins.height
    b = (
        bins.with_row_index("bin_idx")
        .to_pandas()
        .rename(columns={"bin_start": "start", "bin_end": "end"})
    )
    b["chrom"] = b["chrom"].astype(str)
    i = all_cres.to_pandas()
    i["chrom"] = i["chrom"].astype(str)
    joined = bf.overlap(b, i, how="left", suffixes=("", "_cre"))
    out: list[str | None] = [None] * bins.height
    grouped = (
        joined.dropna(subset=["cre_class_cre"])
        .groupby("bin_idx")["cre_class_cre"]
        .agg(lambda s: ",".join(sorted(set(s))))
    )
    for idx, label in grouped.items():
        out[int(idx)] = label
    return out


def _mean_bigwig_score(
    bins: pl.DataFrame, bw_path: str, chrom_prefix: str = "chr"
) -> np.ndarray:
    """Compute mean bigwig score per bin. NaNs in the bigwig become 0 for the
    purposes of the mean (matches ``pyBigWig.stats`` default).
    """
    if bins.height == 0:
        return np.zeros(0, dtype=float)
    bw = pyBigWig.open(bw_path)
    try:
        chroms = bins["chrom"].to_list()
        starts = bins["bin_start"].to_list()
        ends = bins["bin_end"].to_list()
        out = np.empty(len(chroms), dtype=float)
        for i, (c, s, e) in enumerate(zip(chroms, starts, ends)):
            values = bw.values(f"{chrom_prefix}{c}", int(s), int(e), numpy=True)
            values = np.where(np.isnan(values), 0.0, values)
            out[i] = float(values.mean()) if len(values) else 0.0
        return out
    finally:
        bw.close()


def top_misclassified_bins(
    val_predictions: pl.DataFrame,
    *,
    exons_by_species: dict[str, pl.DataFrame],
    all_cres_by_species: dict[str, pl.DataFrame],
    conservation_bw_by_species: dict[str, str],
    top_k: int = 10,
) -> pl.DataFrame:
    """Return top-k false-positive and false-negative bins per species.

    Args:
        val_predictions: Long-form predictions with columns
            ``genome, chrom, bin_start, bin_end, label, logit``.
            Usually the full-coverage eval output so the ranking is over
            every bin of the held-out chromosome(s), not a random subset.
        exons_by_species: ``{genome: polars DF with (chrom, start, end)}``.
        all_cres_by_species: ``{genome: polars DF with (chrom, start, end, cre_class)}``.
        conservation_bw_by_species: ``{genome: path to phastCons bigwig}``.
        top_k: How many of each error type to keep per species.

    Returns:
        Long-form DataFrame (<= 2 * top_k * n_species rows) with columns:
        ``error_type, genome, chrom, bin_start, bin_end, label, logit,
        probability, overlaps_exon, overlaps_any_cre, overlapping_cre_classes,
        mean_phastcons``. Sorted by ``(genome, error_type, rank)``.
    """
    required = {"genome", "chrom", "bin_start", "bin_end", "label", "logit"}
    missing = required - set(val_predictions.columns)
    assert not missing, f"val_predictions missing columns: {missing}"

    parts: list[pl.DataFrame] = []
    for genome, group in val_predictions.group_by("genome", maintain_order=True):
        genome = str(genome[0])
        fp = (
            group.filter(pl.col("label") == 0)
            .sort("logit", descending=True)
            .head(top_k)
            .with_columns(error_type=pl.lit("false_positive"))
        )
        fn = (
            group.filter(pl.col("label") == 1)
            .sort("logit", descending=False)
            .head(top_k)
            .with_columns(error_type=pl.lit("false_negative"))
        )
        top = pl.concat([fp, fn])
        if top.height == 0:
            continue

        exons = exons_by_species.get(genome, pl.DataFrame(schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64}))
        all_cres = all_cres_by_species.get(
            genome,
            pl.DataFrame(
                schema={
                    "chrom": pl.Utf8,
                    "start": pl.Int64,
                    "end": pl.Int64,
                    "cre_class": pl.Utf8,
                }
            ),
        )

        overlaps_exon = _overlap_flags(top, exons)
        cre_classes = _overlap_cre_classes(top, all_cres)
        overlaps_any_cre = np.array([c is not None for c in cre_classes], dtype=bool)
        mean_cons = (
            _mean_bigwig_score(top, conservation_bw_by_species[genome])
            if genome in conservation_bw_by_species
            else np.full(top.height, np.nan)
        )

        annotated = top.with_columns(
            probability=pl.Series(_prob_from_logit(top["logit"].to_numpy())),
            overlaps_exon=pl.Series(overlaps_exon),
            overlaps_any_cre=pl.Series(overlaps_any_cre),
            overlapping_cre_classes=pl.Series(cre_classes, dtype=pl.Utf8),
            mean_phastcons=pl.Series(mean_cons),
        )
        parts.append(annotated)

    if not parts:
        return pl.DataFrame(
            schema={
                "error_type": pl.Utf8,
                "genome": pl.Utf8,
                "chrom": pl.Utf8,
                "bin_start": pl.Int64,
                "bin_end": pl.Int64,
                "label": pl.UInt8,
                "logit": pl.Float32,
                "probability": pl.Float64,
                "overlaps_exon": pl.Boolean,
                "overlaps_any_cre": pl.Boolean,
                "overlapping_cre_classes": pl.Utf8,
                "mean_phastcons": pl.Float64,
            }
        )

    result = pl.concat(parts)
    keep = [
        "error_type",
        "genome",
        "chrom",
        "bin_start",
        "bin_end",
        "label",
        "logit",
        "probability",
        "overlaps_exon",
        "overlaps_any_cre",
        "overlapping_cre_classes",
        "mean_phastcons",
    ]
    return result.select([c for c in keep if c in result.columns])

"""Diagnostics for comparing two interval recipes.

Built to investigate why segmentation-based recipe v20 underperformed
projection-based v30 on TraitGym Mendelian v2 distal AUPRC (issue #136).
"""

import bioframe as bf
import numpy as np
import pandas as pd
import polars as pl
import py2bit
import pyBigWig

from bolinas.data.bin_predictions import top_quantile_bins_to_windows
from bolinas.data.intervals import GenomicSet


def interval_recall(query: GenomicSet, reference: GenomicSet) -> float:
    """Fraction of `reference` intervals overlapped by at least one `query` interval.

    Asymmetric: `interval_recall(A, B)` = recall of B given query A.
    Operates on the merged interval representation of both sets.
    """
    n_total = reference.n_intervals()
    if n_total == 0:
        return 0.0
    n_missed = reference.filter_not_overlapping(query).n_intervals()
    return (n_total - n_missed) / n_total


def bp_jaccard(a: GenomicSet, b: GenomicSet) -> float:
    """Jaccard index over base pairs: |A ∩ B| / |A ∪ B|.

    Returns 0.0 when both sets are empty.
    """
    union_bp = (a | b).total_size()
    if union_bp == 0:
        return 0.0
    inter_bp = (a & b).total_size()
    return inter_bp / union_bp


def softmask_fraction(intervals: GenomicSet, twobit_path: str) -> pd.Series:
    """Per-interval lowercase-base fraction (fraction of soft-masked bp).

    Returns a Series aligned to ``intervals.to_pandas()`` row order.
    """
    df = intervals.to_pandas()
    if len(df) == 0:
        return pd.Series([], dtype=float)
    tb = py2bit.open(twobit_path, storeMasked=True)
    try:
        fracs = np.empty(len(df), dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            seq = tb.sequence(row["chrom"], int(row["start"]), int(row["end"]))
            n = len(seq)
            if n == 0:
                fracs[i] = float("nan")
                continue
            arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            n_lower = int(((arr >= ord("a")) & (arr <= ord("z"))).sum())
            fracs[i] = n_lower / n
    finally:
        tb.close()
    return pd.Series(fracs, index=df.index)


def classify_distal_vs_proximal(
    intervals: GenomicSet, promoters: GenomicSet
) -> pd.Series:
    """Boolean Series: True if interval midpoint is NOT inside any promoter window.

    "Distal" is defined here as midpoint-not-in-promoter, mirroring the
    convention used in the dataset_creation pipeline. Aligned to
    ``intervals.to_pandas()`` row order.
    """
    df = intervals.to_pandas()
    if len(df) == 0:
        return pd.Series([], dtype=bool)
    if promoters.n_intervals() == 0:
        return pd.Series([True] * len(df), index=df.index)
    mid = ((df["start"] + df["end"]) // 2).astype(int)
    midpoints = pd.DataFrame(
        {
            "chrom": df["chrom"].values,
            "start": mid.values,
            "end": (mid + 1).values,
        }
    )
    counts = bf.count_overlaps(midpoints, promoters.to_pandas())
    return pd.Series((counts["count"] == 0).values, index=df.index)


def per_interval_bigwig_aggregates(
    intervals: GenomicSet,
    bigwig_path: str,
    threshold: float,
    chrom_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Aggregate bigwig stats over intervals.

    NaN positions in the bigwig are treated as "not conserved" — they count
    toward total bp but not toward `>= threshold` bp; mean is computed
    excluding NaN.

    Args:
        intervals: GenomicSet of intervals to summarize.
        bigwig_path: Path to a .bw file (e.g. phyloP or phastCons).
        threshold: Cutoff for the conserved-fraction metric.
        chrom_map: Optional dict mapping interval chrom names to bigwig chrom
            names (e.g. RefSeq → UCSC). Intervals on chroms missing from the
            map are dropped (and counted in `n_unmapped`).

    Returns:
        Dict with keys:
            mean: Mean bigwig value across all bp (NaN-excluded).
            frac_ge_threshold: Fraction of bp with value >= threshold (of all bp).
            total_bp: Total bp covered by mapped intervals.
            n_unmapped: Number of intervals dropped due to missing chrom mapping.
    """
    df = intervals.to_pandas()
    n_unmapped = 0
    if chrom_map is not None:
        mapped_chrom = df["chrom"].map(chrom_map)
        mask = mapped_chrom.notna()
        n_unmapped = int((~mask).sum())
        df = df.loc[mask].assign(chrom=mapped_chrom[mask].values)

    if len(df) == 0:
        return {
            "mean": float("nan"),
            "frac_ge_threshold": float("nan"),
            "total_bp": 0,
            "n_unmapped": n_unmapped,
        }

    bw = pyBigWig.open(bigwig_path)
    sum_value = 0.0
    count_non_nan = 0
    count_ge_threshold = 0
    total_bp = 0
    try:
        for _, row in df.iterrows():
            vals = bw.values(
                row["chrom"], int(row["start"]), int(row["end"]), numpy=True
            )
            vals = np.asarray(vals, dtype=np.float32)
            n = vals.size
            non_nan_mask = ~np.isnan(vals)
            n_non_nan = int(non_nan_mask.sum())
            if n_non_nan > 0:
                sum_value += float(vals[non_nan_mask].sum())
                count_non_nan += n_non_nan
                count_ge_threshold += int((vals[non_nan_mask] >= threshold).sum())
            total_bp += int(n)
    finally:
        bw.close()

    return {
        "mean": (sum_value / count_non_nan) if count_non_nan > 0 else float("nan"),
        "frac_ge_threshold": (count_ge_threshold / total_bp)
        if total_bp > 0
        else float("nan"),
        "total_bp": total_bp,
        "n_unmapped": n_unmapped,
    }


def compute_recipe_summary(
    *,
    v20_bed: str,
    v30_bed: str,
    twobit: str,
    promoters_parquet: str,
    ccre_paths: dict[str, str] | None = None,
    ccre_chrom_map: dict[str, str] | None = None,
    scannable_bed: str | None = None,
    exons_parquet: str | None = None,
    conservation_tracks: dict[str, tuple[str, float]] | None = None,
    chrom_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run the full recipe-vs-recipe diagnostic battery for one species.

    Args:
        v20_bed: Path to recipe v20 BED for this species.
        v30_bed: Path to recipe v30 BED for this species.
        twobit: Path to soft-masked 2bit for this species.
        promoters_parquet: Path to promoter regions parquet (used to classify
            distal vs proximal by interval midpoint).
        ccre_paths: Optional dict {label: path_to_cre_parquet}. For each label,
            interval-recall (cCREs hit by recipe) and interval-precision (recipe
            intervals hitting cCREs) are computed. If None, cCRE metrics are
            skipped (e.g. no mouse cCRE source available locally).
        ccre_chrom_map: Optional dict mapping cCRE chrom names to recipe chrom
            names (e.g. bare digit '1' → RefSeq 'NC_000001.11'). Required when
            cCRE and recipe BEDs use different chrom naming conventions; cCREs
            on chroms missing from the map are dropped.
        scannable_bed: Optional path to scannable regions BED. When set together
            with ccre_paths, additionally compute cre_recall_{label}_in_scannable
            against (cCRE ∩ scannable) — the actual upper bound recipes can
            achieve, since both subtract / clip exons via the same scannable mask.
        exons_parquet: Optional path to exons parquet. When set, compute
            frac_bp_overlap_exons per recipe — should be ~0 by construction
            since both recipes subtract the same exon set.
        conservation_tracks: Optional dict {label: (bigwig_path, threshold)}.
            For each track, mean and frac-≥-threshold are computed for each
            recipe. If None, conservation metrics are skipped.
        chrom_map: Optional dict mapping recipe-BED chrom names to bigwig chrom
            names (e.g. RefSeq → UCSC). Required when conservation_tracks is set.

    Returns:
        Tidy DataFrame with columns ['recipe', 'metric', 'value'].
    """
    rows: list[dict] = []

    v20 = GenomicSet.read_bed(v20_bed)
    v30 = GenomicSet.read_bed(v30_bed)
    promoters = GenomicSet.read_parquet(promoters_parquet)
    scannable = GenomicSet.read_bed(scannable_bed) if scannable_bed else None
    exons = GenomicSet.read_parquet(exons_parquet) if exons_parquet else None

    ccre_sets: dict[str, GenomicSet] = {}
    ccre_in_scannable_sets: dict[str, GenomicSet] = {}
    if ccre_paths:
        for ccre_label, ccre_path in ccre_paths.items():
            ccre_df = pd.read_parquet(ccre_path)
            if ccre_chrom_map is not None:
                mapped = ccre_df["chrom"].astype(str).map(ccre_chrom_map)
                n_dropped = int(mapped.isna().sum())
                ccre_df = ccre_df.loc[mapped.notna()].assign(
                    chrom=mapped.dropna().values
                )
                if n_dropped > 0:
                    rows.append(
                        {
                            "recipe": "info",
                            "metric": f"cre_n_dropped_{ccre_label}_unmapped_chrom",
                            "value": float(n_dropped),
                        }
                    )
            ccre = GenomicSet(ccre_df)
            ccre_sets[ccre_label] = ccre
            if scannable is not None:
                ccre_in_scannable_sets[ccre_label] = ccre & scannable

    for name, recipe in [("v20", v20), ("v30", v30)]:
        rows.append(
            {
                "recipe": name,
                "metric": "n_intervals",
                "value": float(recipe.n_intervals()),
            }
        )
        rows.append(
            {"recipe": name, "metric": "total_bp", "value": float(recipe.total_size())}
        )

        if ccre_sets:
            for ccre_label, ccre in ccre_sets.items():
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"cre_recall_{ccre_label}",
                        "value": interval_recall(query=recipe, reference=ccre),
                    }
                )
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"cre_precision_{ccre_label}",
                        "value": interval_recall(query=ccre, reference=recipe),
                    }
                )
                if ccre_label in ccre_in_scannable_sets:
                    ccre_scn = ccre_in_scannable_sets[ccre_label]
                    rows.append(
                        {
                            "recipe": name,
                            "metric": f"cre_recall_{ccre_label}_in_scannable",
                            "value": interval_recall(query=recipe, reference=ccre_scn),
                        }
                    )

        if exons is not None:
            recipe_total = recipe.total_size()
            recipe_in_exons = (recipe & exons).total_size()
            rows.append(
                {
                    "recipe": name,
                    "metric": "frac_bp_overlap_exons",
                    "value": (recipe_in_exons / recipe_total)
                    if recipe_total > 0
                    else float("nan"),
                }
            )

        is_distal = classify_distal_vs_proximal(recipe, promoters)
        rows.append(
            {
                "recipe": name,
                "metric": "frac_distal",
                "value": float(is_distal.mean())
                if len(is_distal) > 0
                else float("nan"),
            }
        )

        softmask = softmask_fraction(recipe, twobit)
        rows.append(
            {
                "recipe": name,
                "metric": "mean_softmask_frac",
                "value": float(softmask.mean()) if len(softmask) > 0 else float("nan"),
            }
        )

        if conservation_tracks:
            for cons_label, (bw_path, threshold) in conservation_tracks.items():
                stats = per_interval_bigwig_aggregates(
                    recipe, bw_path, threshold=threshold, chrom_map=chrom_map
                )
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"mean_{cons_label}",
                        "value": stats["mean"],
                    }
                )
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"frac_{cons_label}_ge_threshold",
                        "value": stats["frac_ge_threshold"],
                    }
                )

    rows.append(
        {"recipe": "v20_v30", "metric": "bp_jaccard", "value": bp_jaccard(v20, v30)}
    )
    rows.append(
        {
            "recipe": "v20_v30",
            "metric": "interval_recall_v20_in_v30",
            "value": interval_recall(query=v20, reference=v30),
        }
    )
    rows.append(
        {
            "recipe": "v20_v30",
            "metric": "interval_recall_v30_in_v20",
            "value": interval_recall(query=v30, reference=v20),
        }
    )

    return pd.DataFrame(rows)


def compute_seg_quantile_sweep(
    *,
    predictions_parquet: str,
    exons_parquet: str,
    defined_bed: str,
    quantiles: list[float],
    target_size: int,
    ccre_paths: dict[str, str] | None = None,
    ccre_chrom_map: dict[str, str] | None = None,
    scannable_bed: str | None = None,
    promoters_parquet: str | None = None,
    twobit: str | None = None,
    conservation_tracks: dict[str, tuple[str, float]] | None = None,
    chrom_map: dict[str, str] | None = None,
    v30_bed: str | None = None,
) -> pd.DataFrame:
    """Sweep top-quantile segmentation thresholds; for each, compute the same
    metric battery as `compute_recipe_summary`.

    Mirrors the recipe-v20 transformation
    (`(top_quantile_bins_to_windows(predictions, q, target_size) - exons) & defined`)
    at multiple quantiles to characterize the threshold's impact on cCRE
    recall, precision, conservation, etc.

    Args:
        predictions_parquet: Per-bin segmentation logits (output of
            `predict_enhancers_segmentation`).
        exons_parquet: Exons to subtract (matches v20 rule input).
        defined_bed: Defined-regions BED to clip to (matches v20 rule input).
        quantiles: List of top-quantile values in (0, 1] to sweep.
        target_size: Output window size in bp (255 in v20).
        ccre_paths, ccre_chrom_map, scannable_bed, promoters_parquet, twobit,
            conservation_tracks, chrom_map: see `compute_recipe_summary`.
        v30_bed: Optional path to recipe v30 BED. When set, the same metric
            battery is also computed against v30 with `recipe="v30_reference"`,
            for direct comparison.

    Returns:
        Tidy DataFrame with columns ['recipe', 'metric', 'value']. Per-quantile
        recipes are named e.g. 'v20_q1pct', 'v20_q5pct'.
    """
    rows: list[dict] = []

    bin_logits = pl.read_parquet(predictions_parquet)
    exons = GenomicSet.read_parquet(exons_parquet)
    defined = GenomicSet.read_bed(defined_bed)
    promoters = (
        GenomicSet.read_parquet(promoters_parquet) if promoters_parquet else None
    )
    scannable = GenomicSet.read_bed(scannable_bed) if scannable_bed else None

    ccre_sets: dict[str, GenomicSet] = {}
    ccre_in_scannable_sets: dict[str, GenomicSet] = {}
    if ccre_paths:
        for label, path in ccre_paths.items():
            ccre_df = pd.read_parquet(path)
            if ccre_chrom_map is not None:
                mapped = ccre_df["chrom"].astype(str).map(ccre_chrom_map)
                ccre_df = ccre_df.loc[mapped.notna()].assign(
                    chrom=mapped.dropna().values
                )
            ccre = GenomicSet(ccre_df)
            ccre_sets[label] = ccre
            if scannable is not None:
                ccre_in_scannable_sets[label] = ccre & scannable

    def _emit_metrics(name: str, recipe: GenomicSet) -> None:
        rows.append(
            {
                "recipe": name,
                "metric": "n_intervals",
                "value": float(recipe.n_intervals()),
            }
        )
        rows.append(
            {"recipe": name, "metric": "total_bp", "value": float(recipe.total_size())}
        )
        for label, ccre in ccre_sets.items():
            rows.append(
                {
                    "recipe": name,
                    "metric": f"cre_recall_{label}",
                    "value": interval_recall(query=recipe, reference=ccre),
                }
            )
            rows.append(
                {
                    "recipe": name,
                    "metric": f"cre_precision_{label}",
                    "value": interval_recall(query=ccre, reference=recipe),
                }
            )
            if label in ccre_in_scannable_sets:
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"cre_recall_{label}_in_scannable",
                        "value": interval_recall(
                            query=recipe, reference=ccre_in_scannable_sets[label]
                        ),
                    }
                )
        if promoters is not None:
            is_distal = classify_distal_vs_proximal(recipe, promoters)
            rows.append(
                {
                    "recipe": name,
                    "metric": "frac_distal",
                    "value": float(is_distal.mean())
                    if len(is_distal) > 0
                    else float("nan"),
                }
            )
        if twobit is not None:
            sm = softmask_fraction(recipe, twobit)
            rows.append(
                {
                    "recipe": name,
                    "metric": "mean_softmask_frac",
                    "value": float(sm.mean()) if len(sm) > 0 else float("nan"),
                }
            )
        if conservation_tracks:
            for cons_label, (bw_path, threshold) in conservation_tracks.items():
                stats = per_interval_bigwig_aggregates(
                    recipe, bw_path, threshold=threshold, chrom_map=chrom_map
                )
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"mean_{cons_label}",
                        "value": stats["mean"],
                    }
                )
                rows.append(
                    {
                        "recipe": name,
                        "metric": f"frac_{cons_label}_ge_threshold",
                        "value": stats["frac_ge_threshold"],
                    }
                )

    for q in quantiles:
        windows = top_quantile_bins_to_windows(
            bin_logits, top_quantile=q, target_size=target_size
        )
        windows = (windows - exons) & defined
        # 0.01 -> "1pct", 0.025 -> "2p5pct"
        pct = q * 100
        if pct == int(pct):
            label = f"{int(pct)}pct"
        else:
            label = f"{pct:g}".replace(".", "p") + "pct"
        _emit_metrics(f"v20_q{label}", windows)

    if v30_bed:
        v30 = GenomicSet.read_bed(v30_bed)
        _emit_metrics("v30_reference", v30)

    return pd.DataFrame(rows)


def compute_disjoint_subsets_summary(
    *,
    v20_bed: str,
    v30_bed: str,
    twobit: str,
    promoters_parquet: str,
    cre_all_parquet: str | None = None,
    ccre_chrom_map: dict[str, str] | None = None,
    conservation_tracks: dict[str, tuple[str, float]] | None = None,
    chrom_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Partition v20 and v30 into shared / unique subsets at the interval
    level and compute the same metric battery for each.

    Subsets emitted (at the interval level — whole v20/v30 intervals are
    kept-or-dropped, never split):

      * v20_total, v30_total: full sets (sanity reference)
      * v20_only:    v20 intervals overlapping no v30 interval
      * v20_shared:  v20 intervals overlapping ≥1 v30 interval
      * v30_only:    v30 intervals overlapping no v20 interval
      * v30_shared:  v30 intervals overlapping ≥1 v20 interval

    Per subset: n_intervals, total_bp, frac_distal, mean_softmask_frac,
    conservation (mean + frac ≥ threshold per track), and — if
    `cre_all_parquet` is set — per-cCRE-class overlap fractions
    (`frac_overlap_cre_<class>`), where each value is the fraction of subset
    intervals overlapping ≥1 cCRE of that class.
    """
    rows: list[dict] = []

    v20 = GenomicSet.read_bed(v20_bed)
    v30 = GenomicSet.read_bed(v30_bed)
    promoters = GenomicSet.read_parquet(promoters_parquet)

    v20_df = v20.to_pandas()
    v30_df = v30.to_pandas()
    counts_v20_in_v30 = bf.count_overlaps(v20_df, v30_df)["count"].to_numpy()
    counts_v30_in_v20 = bf.count_overlaps(v30_df, v20_df)["count"].to_numpy()

    subsets: dict[str, GenomicSet] = {
        "v20_total": v20,
        "v20_only": GenomicSet(v20_df.loc[counts_v20_in_v30 == 0]),
        "v20_shared": GenomicSet(v20_df.loc[counts_v20_in_v30 > 0]),
        "v30_total": v30,
        "v30_only": GenomicSet(v30_df.loc[counts_v30_in_v20 == 0]),
        "v30_shared": GenomicSet(v30_df.loc[counts_v30_in_v20 > 0]),
    }

    cre_class_sets: dict[str, GenomicSet] = {}
    if cre_all_parquet:
        cre_df = pd.read_parquet(cre_all_parquet)
        if ccre_chrom_map is not None:
            mapped = cre_df["chrom"].astype(str).map(ccre_chrom_map)
            cre_df = cre_df.loc[mapped.notna()].assign(chrom=mapped.dropna().values)
        for cls, cls_df in cre_df.groupby("cre_class", sort=False):
            cre_class_sets[str(cls)] = GenomicSet(cls_df[["chrom", "start", "end"]])
        cre_class_sets["any"] = GenomicSet(cre_df[["chrom", "start", "end"]])

    def _emit(name: str, subset: GenomicSet) -> None:
        rows.append(
            {
                "subset": name,
                "metric": "n_intervals",
                "value": float(subset.n_intervals()),
            }
        )
        rows.append(
            {"subset": name, "metric": "total_bp", "value": float(subset.total_size())}
        )
        is_distal = classify_distal_vs_proximal(subset, promoters)
        rows.append(
            {
                "subset": name,
                "metric": "frac_distal",
                "value": float(is_distal.mean())
                if len(is_distal) > 0
                else float("nan"),
            }
        )
        sm = softmask_fraction(subset, twobit)
        rows.append(
            {
                "subset": name,
                "metric": "mean_softmask_frac",
                "value": float(sm.mean()) if len(sm) > 0 else float("nan"),
            }
        )
        if conservation_tracks:
            for cons_label, (bw_path, threshold) in conservation_tracks.items():
                stats = per_interval_bigwig_aggregates(
                    subset, bw_path, threshold=threshold, chrom_map=chrom_map
                )
                rows.append(
                    {
                        "subset": name,
                        "metric": f"mean_{cons_label}",
                        "value": stats["mean"],
                    }
                )
                rows.append(
                    {
                        "subset": name,
                        "metric": f"frac_{cons_label}_ge_threshold",
                        "value": stats["frac_ge_threshold"],
                    }
                )
        for cls_label, cre_cls in cre_class_sets.items():
            rows.append(
                {
                    "subset": name,
                    "metric": f"frac_overlap_cre_{cls_label}",
                    "value": interval_recall(query=cre_cls, reference=subset),
                }
            )

    for name, subset in subsets.items():
        _emit(name, subset)

    return pd.DataFrame(rows)

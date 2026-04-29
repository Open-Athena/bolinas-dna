"""Diagnostics for comparing two interval recipes.

Built to investigate why segmentation-based recipe v20 underperformed
projection-based v30 on TraitGym Mendelian v2 distal AUPRC (issue #136).
"""

import bioframe as bf
import numpy as np
import pandas as pd
import py2bit
import pyBigWig

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
    tb = py2bit.open(twobit_path)
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

    ccre_sets: dict[str, GenomicSet] = {}
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
            ccre_sets[ccre_label] = GenomicSet(ccre_df)

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

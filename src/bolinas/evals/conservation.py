"""Per-variant conservation scoring via UCSC bigWig tracks.

Used by ``snakemake/conservation_eval/`` (issue #146) to score TraitGym
Mendelian v2 variants with classical conservation tracks:

- ``phyloP_100v``    — UCSC 100-vertebrate phyloP (multiz alignment)
- ``phastCons_100v`` — UCSC 100-vertebrate phastCons (multiz alignment)
- ``phyloP_241m``    — Zoonomia 241-mammal Cactus phyloP
- ``phyloP_447m``    — UCSC 447-way phyloP (Zoonomia + densely-sampled primates, Cactus)
- ``phyloP_470m``    — UCSC 470-way phyloP (newest mammal Cactus)
- ``phastCons_470m`` — UCSC 470-way phastCons (newest mammal Cactus)
- ``phastCons_43p``  — Zoonomia 43-primate track (TraitGym name; underlying file is phyloP-over-primates)

The first three tracks (``phyloP_100v``, ``phyloP_241m``, ``phastCons_43p``)
are the original TraitGym set; their URLs are copied verbatim from
TraitGym's ``eval/workflow/rules/conservation.smk``. The same bigWigs are
also used elsewhere in this repo (enhancer_classification,
training_dataset/dataset_creation) but each pipeline manages its own
download to avoid coupling.

NaN policy: this module preserves NaNs from the bigWig (no alignment at that
locus). Callers decide how to fill them — the ``conservation_eval`` pipeline
applies ``fillna(0)`` only at the metrics-aggregation step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from bolinas.evals.metrics import compute_metrics


CONSERVATION_TRACKS: dict[str, str] = {
    # 100-vertebrate UCSC multiz alignment.
    "phyloP_100v": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
    "phastCons_100v": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
    # Zoonomia 241-mammal Cactus alignment.
    "phyloP_241m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw",
    # UCSC 447-way Cactus (Zoonomia + densely-sampled primates, Kuderna et al. 2023).
    "phyloP_447m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw",
    # UCSC 470-way Cactus (newest mammal alignment).
    "phyloP_470m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/hg38.phyloP470way.bw",
    "phastCons_470m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw",
    # phastCons_43p name is a TraitGym convention; the underlying file is
    # actually phyloP over 43 primates from the Zoonomia track hub. We keep
    # the name to stay consistent with TraitGym + existing config in this repo.
    "phastCons_43p": "https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig",
}


# TraitGym variant columns the pipeline preserves end-to-end. Asserted by
# the score and aggregate stages so a schema drift fails fast.
REQUIRED_VARIANT_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
)


def score_variants_at_positions(
    df: pd.DataFrame,
    bw_path: str | Path,
) -> np.ndarray:
    """Look up a bigWig value for each variant in ``df``.

    TraitGym variants are 1-based (VCF convention); pyBigWig is 0-based
    half-open. The conversion ``[pos - 1, pos)`` selects the single base at
    1-based ``pos``.

    NaNs are preserved: pyBigWig returns ``nan`` at positions with no data.

    Args:
        df: Must have integer ``pos`` and string ``chrom`` columns. ``chrom``
            entries without a ``chr`` prefix are auto-prefixed to match UCSC
            bigWig naming.
        bw_path: Path to the bigWig file.

    Returns:
        ``np.ndarray`` of shape ``(len(df),)`` with one float per row.
    """
    assert "chrom" in df.columns and "pos" in df.columns, (
        "df must have chrom + pos columns"
    )
    assert pd.api.types.is_integer_dtype(df["pos"]), (
        f"pos must be integer dtype, got {df['pos'].dtype}"
    )

    bw = pyBigWig.open(str(bw_path))
    try:
        bw_chroms = set(bw.chroms())
        scores = np.empty(len(df), dtype=np.float64)
        for i, (chrom, pos) in enumerate(zip(df["chrom"], df["pos"])):
            chrom_str = str(chrom)
            if not chrom_str.startswith("chr"):
                chrom_str = f"chr{chrom_str}"
            if chrom_str not in bw_chroms:
                # No track for this chromosome (e.g. patches, alt contigs).
                scores[i] = np.nan
                continue
            # 1-based pos -> 0-based half-open [pos-1, pos): one base.
            vals = bw.values(chrom_str, int(pos) - 1, int(pos))
            scores[i] = vals[0] if vals else np.nan
    finally:
        bw.close()

    return scores


def aggregate_traitgym_metrics(
    parquet_paths: dict[str, str | Path],
) -> tuple[pd.DataFrame, str]:
    """Aggregate per-score scored-variant parquets into a metrics DataFrame
    and a markdown report.

    Each input parquet is the output of ``score_variants_at_positions`` plus
    the original TraitGym columns: must contain ``[chrom, pos, ref, alt,
    label, subset, score]``. The ``score`` column may contain NaN (positions
    with no alignment in the bigWig).

    For each score: NaN count is recorded per subset, then ``score`` is
    filled with 0 (semantically meaningful — see module docstring) before
    AUPRC is computed via ``bolinas.evals.metrics.compute_metrics``.

    Args:
        parquet_paths: mapping ``score_name -> parquet path``. Order is
            preserved in the markdown table.

    Returns:
        ``(metrics_df, markdown)`` where ``metrics_df`` has columns
        ``[metric, score_type, score_name, subset, value, n_pos, n_neg,
        n_nan, n_total]``.
    """
    assert parquet_paths, "parquet_paths must be non-empty"

    score_names = list(parquet_paths)
    required = (*REQUIRED_VARIANT_COLUMNS, "score")
    all_metrics: list[pd.DataFrame] = []

    for score_name in score_names:
        df = pd.read_parquet(parquet_paths[score_name])
        for col in required:
            assert col in df.columns, (
                f"{parquet_paths[score_name]}: missing column {col!r}"
            )

        # Count NaN per subset before filling; "global" matches compute_metrics'
        # convention for the all-rows row.
        nan_per_subset = df.groupby("subset")["score"].apply(
            lambda s: int(s.isna().sum())
        )
        nan_per_subset["global"] = int(df["score"].isna().sum())
        total_per_subset = df.groupby("subset").size().astype(int)
        total_per_subset["global"] = len(df)

        m = compute_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[["score"]].fillna(0),
            metrics=["AUPRC"],
            score_columns=["score"],
        )
        m["score_name"] = score_name
        m["n_nan"] = m["subset"].map(nan_per_subset).astype(int)
        m["n_total"] = m["subset"].map(total_per_subset).astype(int)
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    md = _build_markdown(metrics, score_names)
    return metrics, md


def _build_markdown(metrics: pd.DataFrame, score_names: list[str]) -> str:
    """Render the metrics DataFrame as a two-table markdown report.

    AUPRC table: per-subset rows plus an unweighted ``mean`` row across
    subsets at the top (macro-AUPRC). The ``global`` row is intentionally
    excluded — it would be dominated by the largest subset (missense).

    NaN-counts table: keeps the ``global`` row (it's a real total count,
    not an aggregate of AUPRCs).
    """
    auprc = metrics[metrics["metric"] == "AUPRC"].copy()

    def _pivot(values_col: str) -> pd.DataFrame:
        return auprc.pivot_table(
            index="subset",
            columns="score_name",
            values=values_col,
            aggfunc="first",
        )

    # Per-subset n_pos/n_neg/n_total from the first score (subset coverage
    # is score-independent).
    coverage = (
        auprc[auprc["score_name"] == score_names[0]][
            ["subset", "n_pos", "n_neg", "n_total"]
        ]
        .drop_duplicates(subset="subset")
        .set_index("subset")
    )

    pivot = _pivot("value")
    nan_pivot = _pivot("n_nan")

    # Per-subset rows ordered by n_pos descending; "global" is excluded
    # from the AUPRC table but kept for the NaN-counts table.
    per_subset = [
        s for s in coverage.sort_values("n_pos", ascending=False).index if s != "global"
    ]
    pivot_subsets = pivot.reindex(per_subset)

    # Unweighted mean of per-subset AUPRCs (one value per score). NaN-skip
    # so a missing subset value for one score doesn't take down the mean.
    mean_row = pivot_subsets.mean(axis=0, skipna=True)

    lines: list[str] = []
    lines.append("### TraitGym Mendelian v2 — AUPRC")
    lines.append("")
    lines.append(
        "Per-subset AUPRC. The top `mean` row is the unweighted mean across "
        "subsets (macro-AUPRC); each subset contributes equally regardless "
        "of its size."
    )
    lines.append("")
    header = ["subset", "n_pos", "n_neg", *score_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    mean_vals = [
        f"{mean_row[s]:.3f}" if pd.notna(mean_row[s]) else "—" for s in score_names
    ]
    lines.append("| " + " | ".join(["mean", "—", "—", *mean_vals]) + " |")

    for subset in per_subset:
        n_pos = int(coverage.loc[subset, "n_pos"])
        n_neg = int(coverage.loc[subset, "n_neg"])
        vals = [
            f"{pivot.loc[subset, s]:.3f}" if pd.notna(pivot.loc[subset, s]) else "—"
            for s in score_names
        ]
        lines.append("| " + " | ".join([subset, str(n_pos), str(n_neg), *vals]) + " |")

    # NaN counts: keep the global row at the top, then per-subset rows.
    nan_order = (["global"] if "global" in nan_pivot.index else []) + per_subset
    nan_pivot_ordered = nan_pivot.reindex(nan_order)

    lines.append("")
    lines.append("### NaN counts")
    lines.append("")
    lines.append(
        "NaN = no alignment at that locus in the bigWig. AUPRC above is computed "
        "after `.fillna(0)`: 0 is semantically meaningful for both phyloP "
        "(neither conserved nor accelerated) and phastCons (non-conserved)."
    )
    lines.append("")
    nan_header = ["subset", "n_total", *score_names]
    lines.append("| " + " | ".join(nan_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(nan_header)) + " |")
    for subset in nan_pivot_ordered.index:
        n_total = (
            int(coverage.loc[subset, "n_total"]) if subset in coverage.index else 0
        )
        vals = [str(int(nan_pivot_ordered.loc[subset, s])) for s in score_names]
        lines.append("| " + " | ".join([subset, str(n_total), *vals]) + " |")

    return "\n".join(lines) + "\n"

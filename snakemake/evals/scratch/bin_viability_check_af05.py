"""Same viability check as bin_viability_check.py, but with negatives
filtered to AF > 0.05 on the fly — simulates the AF>5% common threshold
without rebuilding gnomad_common.
"""

from __future__ import annotations

import polars as pl

DATASET_ALL = "s3://oa-bolinas/snakemake/evals/results/mendelian_traits/dataset_all.parquet"
OUT = "s3://oa-bolinas/snakemake/evals/results/scratch/bin_viability_check_af05.parquet"
K = 9
NEG_AF_MIN = 0.05

SCHEMES: list[tuple[str, list[tuple[str, list[float]]]]] = [
    (
        "tss_proximal",
        [
            ("distance_tss_pc", [0.0, 100.0, 1000.0, float("inf")]),
            ("distance_exon_pc", [0.0, 100.0, 1000.0, float("inf")]),
        ],
    ),
    ("splicing", [("distance_exon_pc", [0.0, 5.0, 30.0, float("inf")])]),
    ("distal", [("distance_exon_pc", [0.0, 1000.0, 10000.0, float("inf")])]),
]


def bin_label(col: pl.Expr, edges: list[float]) -> pl.Expr:
    expr = pl.lit("OOR")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        cond = (col >= lo) & (col < hi)
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return expr


def main() -> None:
    print(f"Reading {DATASET_ALL} …", flush=True)
    da = pl.read_parquet(DATASET_ALL)
    n_neg_pre = da.filter(~pl.col("label")).height
    da = da.filter(pl.col("label") | (pl.col("AF") > NEG_AF_MIN))
    n_neg_post = da.filter(~pl.col("label")).height
    print(
        f"  rows={da.height:,}  (negatives after AF>{NEG_AF_MIN} filter: "
        f"{n_neg_post:,} of {n_neg_pre:,} = {n_neg_post / n_neg_pre:.1%})",
        flush=True,
    )

    out_rows: list[dict] = []
    for subset, specs in SCHEMES:
        sub = da.filter(pl.col("consequence_group") == subset)
        n_pos_total = sub.filter(pl.col("label")).height
        n_neg_total = sub.filter(~pl.col("label")).height
        print(
            f"\n=== {subset}  pool pos={n_pos_total:,}  neg={n_neg_total:,} ===",
            flush=True,
        )
        for feat, edges in specs:
            print(f"  binning {feat} with edges {edges}", flush=True)

        bin_cols = []
        for feat, edges in specs:
            col_name = f"{feat}_bin"
            sub = sub.with_columns(bin_label(pl.col(feat), edges).alias(col_name))
            bin_cols.append(col_name)

        group_cols = ["chrom"] + bin_cols
        counts = sub.group_by(group_cols + ["label"]).agg(pl.len().alias("n"))
        wide = counts.pivot(values="n", index=group_cols, on="label").rename(
            {"true": "n_pos", "false": "n_neg"}
        ).fill_null(0)

        strata_with_pos = wide.filter(pl.col("n_pos") > 0)
        strata_with_pos = strata_with_pos.with_columns(
            n_pos_kept=pl.min_horizontal(pl.col("n_pos"), pl.col("n_neg") // K),
        ).with_columns(n_pos_dropped=pl.col("n_pos") - pl.col("n_pos_kept"))

        total_pos = strata_with_pos["n_pos"].sum()
        total_kept = strata_with_pos["n_pos_kept"].sum()
        total_dropped = total_pos - total_kept
        thin = strata_with_pos.filter(pl.col("n_neg") < K * pl.col("n_pos"))
        print(
            f"  strata with positives: {strata_with_pos.height}, "
            f"thin (neg < K*pos): {thin.height}",
            flush=True,
        )
        print(
            f"  positives total={total_pos}  kept={total_kept}  dropped={total_dropped} "
            f"({total_dropped / total_pos:.1%})",
            flush=True,
        )
        if thin.height:
            print("  thin strata (top 10 by n_pos):", flush=True)
            for r in thin.sort("n_pos", descending=True).head(10).iter_rows(named=True):
                bins_str = ", ".join(f"{c}={r[c]}" for c in bin_cols)
                print(
                    f"    chrom={r['chrom']:>3s}  {bins_str}  pos={r['n_pos']:>4d}  "
                    f"neg={r['n_neg']:>6d}  drop={r['n_pos_dropped']:>3d}",
                    flush=True,
                )

        for r in strata_with_pos.iter_rows(named=True):
            out_rows.append(
                {
                    "subset": subset,
                    **{c: r[c] for c in group_cols},
                    "n_pos": r["n_pos"],
                    "n_neg": r["n_neg"],
                    "n_pos_kept": r["n_pos_kept"],
                    "n_pos_dropped": r["n_pos_dropped"],
                }
            )

    out = pl.DataFrame(out_rows)
    print(f"\nWriting summary to {OUT}", flush=True)
    out.write_parquet(OUT)


if __name__ == "__main__":
    main()

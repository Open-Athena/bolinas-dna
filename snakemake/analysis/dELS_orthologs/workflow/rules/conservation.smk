"""Per-query conservation scoring + recall@k binned by that score.

The per-query statistic is the **fraction of bases in the query dELS interval
whose conservation track value is ≥ the Zoonomia threshold** (2.27 for
phyloP_241m; 0.961 for phastCons_43p — see config comments). This follows
the pattern in `snakemake/enhancer_classification/workflow/rules/data.smk`.

Aligner alignments are unchanged — these rules re-use the existing
`per_query_report.parquet` and add a conservation-axis slice.
"""


rule score_query_conservation:
    """Per-query conservation statistic: #/fraction of bases >= threshold."""
    input:
        queries="results/cre/hg38/query.filtered.parquet",
        bw="results/conservation/hg38/{track}.bw",
    output:
        "results/conservation/hg38/{track}/query_scores.parquet",
    params:
        threshold=lambda wildcards: config["conservation"]["hg38"][wildcards.track][
            "threshold"
        ],
    run:
        import pyBigWig

        df = pl.read_parquet(input.queries)
        bw = pyBigWig.open(input.bw)
        # `pyBigWig.values` returns None for chroms missing from the bigwig
        # and NaN for positions with no coverage. `vals >= threshold` yields
        # False for NaN so those count as non-conserved, matching the
        # `enhancer_classification` pattern.
        try:
            conserved_bases = []
            for row in df.iter_rows(named=True):
                vals = bw.values(row["chrom"], row["start"], row["end"], numpy=True)
                if vals is None:
                    conserved_bases.append(0)
                else:
                    conserved_bases.append(int(np.sum(vals >= params.threshold)))
        finally:
            bw.close()
        out = df.with_columns(
            pl.Series("conserved_bases", conserved_bases, dtype=pl.Int64),
            (pl.col("end") - pl.col("start")).alias("total_bases"),
        ).with_columns(
            (pl.col("conserved_bases") / pl.col("total_bases"))
            .cast(pl.Float64)
            .alias("pct_conserved")
        )
        out.write_parquet(output[0])
        print(
            f"  {wildcards.track} @ threshold={params.threshold}: "
            f"{out.height:,} queries scored; "
            f"mean pct_conserved={out['pct_conserved'].mean():.4f}"
        )


rule recall_by_conservation:
    """Recall@k (1, 5, 10) per equal-frequency bin of `pct_conserved`.

    Bin edges are quantiles of the per-query pct_conserved distribution,
    computed over the post-filter query universe.
    """
    input:
        report="results/eval/{aligner}/per_query_report.parquet",
        scores="results/conservation/hg38/{track}/query_scores.parquet",
    output:
        parquet="results/eval/{aligner}/recall_by_conservation/{track}.parquet",
        tsv="results/eval/{aligner}/recall_by_conservation/{track}.tsv",
    params:
        n_bins=config["n_conservation_bins"],
    run:
        report = pl.read_parquet(input.report)
        scores = pl.read_parquet(input.scores)

        # Bin queries into equal-frequency quantile buckets by pct_conserved.
        # phastCons_43p has many queries at exactly 0% conserved (strict
        # threshold), so multiple quantile edges collide; `allow_duplicates`
        # lets polars handle ties by merging bins, which may give us fewer
        # than n_bins effective buckets.
        n_bins = int(params.n_bins)
        quantiles = list(np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1))
        scores = scores.with_columns(
            pl.col("pct_conserved")
            .qcut(quantiles, allow_duplicates=True, include_breaks=True)
            .alias("_bin_struct")
        ).unnest("_bin_struct")
        # qcut with include_breaks returns struct (breakpoint, category).
        # Map category (string) to a stable ordered index.
        bin_order = (
            scores.select("category").unique().sort("category")["category"].to_list()
        )
        bin_to_idx = {b: i for i, b in enumerate(bin_order)}
        scores = scores.with_columns(
            pl.col("category")
            .map_elements(lambda c: bin_to_idx[c], return_dtype=pl.Int64)
            .alias("bin_idx"),
            pl.col("category").alias("bin"),
        )

        # Join the per-query bin label onto the report so bin-aware recall
        # only uses rows that are actually top-k hits (rank not null).
        report_binned = report.join(
            scores.select(["accession", "bin_idx", "bin"]),
            left_on="query",
            right_on="accession",
            how="left",
        )

        rows = []
        for b in sorted(scores["bin_idx"].unique().to_list()):
            queries_in_bin = set(
                scores.filter(pl.col("bin_idx") == b)["accession"].to_list()
            )
            n_q = len(queries_in_bin)
            bin_report = report_binned.filter(pl.col("bin_idx") == b)
            bin_label = bin_order[b]
            for k in (1, 5, 10):
                recovered = (
                    bin_report.filter(
                        pl.col("in_gold_standard")
                        & pl.col("rank").is_not_null()
                        & (pl.col("rank") <= k)
                    )["query"]
                    .unique()
                    .to_list()
                )
                n_hits = len(set(recovered) & queries_in_bin)
                rows.append(
                    {
                        "track": wildcards.track,
                        "bin_idx": b,
                        "bin": bin_label,
                        "k": k,
                        "n_queries": n_q,
                        "n_hits": n_hits,
                        "recall": n_hits / n_q if n_q else float("nan"),
                    }
                )

        out = pl.DataFrame(rows)
        out.write_parquet(output.parquet)
        out.write_csv(output.tsv, separator="\t")
        print(f"\n=== recall@k by {wildcards.track} conservation bin ===")
        print(out.to_pandas().to_string(index=False))


rule recall_by_class_and_conservation_fixed:
    """Recall@k stratified by (hg38 cCRE class × fixed-width conservation bin).

    Bins span [0, w), [w, 2w), ..., [1 - w, 1] where w = `conservation_bin_width`.
    Unlike the quantile-binned report, bin membership depends on absolute
    conservation %, so CA/TF-heavy classes concentrate in the low bins and
    dELS/PLS spread higher up — intentional, for cross-class comparability.
    """
    input:
        report="results/eval/{aligner}/per_query_report.parquet",
        scores="results/conservation/hg38/{track}/query_scores.parquet",
        query="results/cre/hg38/query.filtered.parquet",
    output:
        parquet="results/eval/{aligner}/recall_by_class_and_conservation_fixed/{track}.parquet",
        tsv="results/eval/{aligner}/recall_by_class_and_conservation_fixed/{track}.tsv",
    params:
        bin_width=config["conservation_bin_width"],
    run:
        report = pl.read_parquet(input.report)
        scores = pl.read_parquet(input.scores)
        query = pl.read_parquet(input.query).select(["accession", "cre_class"])

        width = float(params.bin_width)
        n_bins = int(round(1.0 / width))


        def _bin_label(i: int) -> str:
            lo, hi = i * width, (i + 1) * width
            closed_hi = "]" if i == n_bins - 1 else ")"
            return f"[{lo:.2f}, {hi:.2f}{closed_hi}"


        pct_np = scores["pct_conserved"].to_numpy()
        bin_idx = np.clip(np.floor(pct_np / width).astype(int), 0, n_bins - 1)
        scores = scores.with_columns(
            pl.Series("bin_idx", bin_idx, dtype=pl.Int64),
            pl.Series(
                "bin",
                [_bin_label(int(i)) for i in bin_idx],
                dtype=pl.Utf8,
            ),
        ).join(query, on="accession", how="left")

        # Attach bin + class to the report for class-aware top-k counting.
        report_binned = report.join(
            scores.select(["accession", "bin_idx", "bin", "cre_class"]),
            left_on="query",
            right_on="accession",
            how="left",
        )

        classes = sorted({c for c in scores["cre_class"].to_list() if c is not None})
        rows = []
        for cls in [*classes, None]:
            cls_scores = (
                scores if cls is None else scores.filter(pl.col("cre_class") == cls)
            )
            cls_report = (
                report_binned
                if cls is None
                else report_binned.filter(pl.col("cre_class") == cls)
            )
            for b in range(n_bins):
                queries_in_bin = set(
                    cls_scores.filter(pl.col("bin_idx") == b)["accession"].to_list()
                )
                n_q = len(queries_in_bin)
                if n_q == 0:
                    continue
                bin_report = cls_report.filter(pl.col("bin_idx") == b)
                for k in (1, 5, 10):
                    recovered = (
                        bin_report.filter(
                            pl.col("in_gold_standard")
                            & pl.col("rank").is_not_null()
                            & (pl.col("rank") <= k)
                        )["query"]
                        .unique()
                        .to_list()
                    )
                    n_hits = len(set(recovered) & queries_in_bin)
                    rows.append(
                        {
                            "track": wildcards.track,
                            "cre_class": cls if cls is not None else "Overall",
                            "bin_idx": b,
                            "bin": _bin_label(b),
                            "k": k,
                            "n_queries": n_q,
                            "n_hits": n_hits,
                            "recall": n_hits / n_q,
                        }
                    )

        out = pl.DataFrame(rows)
        out.write_parquet(output.parquet)
        out.write_csv(output.tsv, separator="\t")


rule recall_by_class_and_conservation_quantile:
    """Recall@k stratified by (hg38 cCRE class × global conservation quantile bin).

    Bins are computed over the full post-filter query universe, not per-class,
    so class-marginal counts vary across bins (classes with mostly-unconserved
    queries sit in the bottom bins). Used together with the fixed-bin report
    to triangulate: fixed bins give equal-conservation-%-ranges, quantile bins
    give equal-sample-size conservation ranges.
    """
    input:
        report="results/eval/{aligner}/per_query_report.parquet",
        scores="results/conservation/hg38/{track}/query_scores.parquet",
        query="results/cre/hg38/query.filtered.parquet",
    output:
        parquet="results/eval/{aligner}/recall_by_class_and_conservation_quantile/{track}.parquet",
        tsv="results/eval/{aligner}/recall_by_class_and_conservation_quantile/{track}.tsv",
    params:
        n_bins=config["n_conservation_bins"],
    run:
        report = pl.read_parquet(input.report)
        scores = pl.read_parquet(input.scores)
        query = pl.read_parquet(input.query).select(["accession", "cre_class"])

        n_bins = int(params.n_bins)
        quantiles = list(np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1))
        scores = (
            scores.with_columns(
                pl.col("pct_conserved")
                .qcut(quantiles, allow_duplicates=True, include_breaks=True)
                .alias("_bin_struct")
            )
            .unnest("_bin_struct")
            .join(query, on="accession", how="left")
        )

        bin_order = (
            scores.select("category").unique().sort("category")["category"].to_list()
        )
        bin_to_idx = {b: i for i, b in enumerate(bin_order)}
        scores = scores.with_columns(
            pl.col("category")
            .map_elements(lambda c: bin_to_idx[c], return_dtype=pl.Int64)
            .alias("bin_idx"),
            pl.col("category").alias("bin"),
        )

        report_binned = report.join(
            scores.select(["accession", "bin_idx", "bin", "cre_class"]),
            left_on="query",
            right_on="accession",
            how="left",
        )

        classes = sorted({c for c in scores["cre_class"].to_list() if c is not None})
        rows = []
        for cls in [*classes, None]:
            cls_scores = (
                scores if cls is None else scores.filter(pl.col("cre_class") == cls)
            )
            cls_report = (
                report_binned
                if cls is None
                else report_binned.filter(pl.col("cre_class") == cls)
            )
            for b in sorted(scores["bin_idx"].unique().to_list()):
                queries_in_bin = set(
                    cls_scores.filter(pl.col("bin_idx") == b)["accession"].to_list()
                )
                n_q = len(queries_in_bin)
                if n_q == 0:
                    continue
                bin_report = cls_report.filter(pl.col("bin_idx") == b)
                bin_label = bin_order[b]
                for k in (1, 5, 10):
                    recovered = (
                        bin_report.filter(
                            pl.col("in_gold_standard")
                            & pl.col("rank").is_not_null()
                            & (pl.col("rank") <= k)
                        )["query"]
                        .unique()
                        .to_list()
                    )
                    n_hits = len(set(recovered) & queries_in_bin)
                    rows.append(
                        {
                            "track": wildcards.track,
                            "cre_class": cls if cls is not None else "Overall",
                            "bin_idx": b,
                            "bin": bin_label,
                            "k": k,
                            "n_queries": n_q,
                            "n_hits": n_hits,
                            "recall": n_hits / n_q,
                        }
                    )

        out = pl.DataFrame(rows)
        out.write_parquet(output.parquet)
        out.write_csv(output.tsv, separator="\t")

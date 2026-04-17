"""Per-query report: for each hg38 query dELS, the top-k aligner hits in the
mm10 search window, annotated with the overlapping mm10 Registry-V4 cCRE
(any class) and gold-standard membership.

Aligner-agnostic: reads a unified-schema hits TSV produced by any
`align_<tool>.smk` rule at `results/align/{aligner}/hits.tsv`.

The dELS-only filter applies to the hg38 query side only (per the tracking
issue); on the mm10 side we keep cCREs of every class so we can see hits to
e.g. mouse `CA` cCREs that are the orthologous element on the mouse side.
See the issue body for the documented hg38-dELS ↔ mm10-CA classification
asymmetry at ZRS.
"""


rule per_query_report:
    input:
        hits="results/align/{aligner}/hits.tsv",
        query="results/cre/hg38/query.filtered.parquet",
        mm10_cres="results/cre/mm10/cres_window.parquet",
        gold="results/orthologs/hg38_mm10.tsv",
    output:
        parquet="results/eval/{aligner}/per_query_report.parquet",
        tsv="results/eval/{aligner}/per_query_report.tsv",
    params:
        top_k=config["report_top_k"],
    run:
        hits = pl.read_csv(input.hits, separator="\t", has_header=True)
        mm10_cres = pl.read_parquet(input.mm10_cres)

        # Hit ↔ mm10 cCRE intersection (any class, any overlap > 0).
        if hits.height == 0:
            joined = pl.DataFrame(
                schema={
                    "query": pl.Utf8,
                    "score": pl.Int64,
                    "fident": pl.Float64,
                    "evalue": pl.Float64,
                    "qcov": pl.Float64,
                    "tcov": pl.Float64,
                    "rev_strand": pl.Boolean,
                    "hit_chrom": pl.Utf8,
                    "hit_start": pl.Int64,
                    "hit_end": pl.Int64,
                    "mm10_accession": pl.Utf8,
                    "mm10_class": pl.Utf8,
                    "mm10_start": pl.Int64,
                    "mm10_end": pl.Int64,
                }
            )
        else:
            j = bf.overlap(
                hits.to_pandas(),
                mm10_cres.to_pandas(),
                cols1=("hit_chrom", "hit_start", "hit_end"),
                cols2=("chrom", "start", "end"),
                suffixes=("", "_cre"),
                how="left",
            )
            joined = (
                pl.from_pandas(j)
                .rename(
                    {
                        "accession_cre": "mm10_accession",
                        "cre_class_cre": "mm10_class",
                        "start_cre": "mm10_start",
                        "end_cre": "mm10_end",
                    }
                )
                .select(
                    [
                        "query",
                        "score",
                        "fident",
                        "evalue",
                        "qcov",
                        "tcov",
                        "rev_strand",
                        "hit_chrom",
                        "hit_start",
                        "hit_end",
                        "mm10_accession",
                        "mm10_class",
                        "mm10_start",
                        "mm10_end",
                    ]
                )
            )

        gold = pl.read_csv(
            input.gold,
            separator="\t",
            has_header=False,
            new_columns=["hg38_accession", "mm10_accession_gold"],
        )
        query_ids = pl.read_parquet(input.query)["accession"].to_list()

        # Gold-pair lookup vectorized as a left-join on (query, mm10_accession).
        # At 311K queries × top-k hits the prior `map_elements` over millions
        # of rows was the dominant cost; a polars join runs in ≪1 s.
        gold_df = (
            gold.rename(
                {"hg38_accession": "query", "mm10_accession_gold": "mm10_accession"}
            )
            .with_columns(pl.lit(True).alias("in_gold_standard"))
            .unique(subset=["query", "mm10_accession"])
        )
        joined = joined.join(
            gold_df, on=["query", "mm10_accession"], how="left"
        ).with_columns(pl.col("in_gold_standard").fill_null(False))

        # Per-query gold partners, O(N) dict build. Reused below by both the
        # missing-rows loop and the stdout summary — the previous
        # `{m for h, m in gold_pairs if h == q}` scan was O(N·M) and would
        # have taken hours at 311K queries × 537K gold pairs.
        gold_by_query: dict[str, list[str]] = {}
        for h, m in zip(
            gold["hg38_accession"].to_list(),
            gold["mm10_accession_gold"].to_list(),
        ):
            gold_by_query.setdefault(h, []).append(m)

        report = (
            joined.sort(["query", "score"], descending=[False, True])
            .with_columns(
                pl.col("score")
                .rank("ordinal", descending=True)
                .over("query")
                .cast(pl.Int64)
                .alias("rank"),
            )
            .filter(pl.col("rank") <= params.top_k)
            .sort(["query", "rank"])
        )

        # Surface gold partners that the aligner did NOT recover, as
        # null-score rows so the per-query view is honest about misses.
        recovered = set(
            zip(
                report["query"].to_list(),
                report.filter(pl.col("mm10_accession").is_not_null())[
                    "mm10_accession"
                ].to_list(),
            )
        )
        missing_rows = []
        mm10_cre_lookup = {r["accession"]: r for r in mm10_cres.iter_rows(named=True)}
        for q in query_ids:
            for partner in set(gold_by_query.get(q, [])):
                if (q, partner) not in recovered:
                    cre = mm10_cre_lookup.get(partner)
                    missing_rows.append(
                        {
                            "query": q,
                            "score": None,
                            "fident": None,
                            "evalue": None,
                            "qcov": None,
                            "tcov": None,
                            "rev_strand": None,
                            "hit_chrom": None,
                            "hit_start": None,
                            "hit_end": None,
                            "mm10_accession": partner,
                            "mm10_class": cre["cre_class"] if cre else None,
                            "mm10_start": cre["start"] if cre else None,
                            "mm10_end": cre["end"] if cre else None,
                            "in_gold_standard": True,
                            "rank": None,
                        }
                    )
        if missing_rows:
            missing = pl.DataFrame(missing_rows, schema_overrides=report.schema)
            report = pl.concat([report, missing], how="diagonal").sort(
                ["query", "rank"], nulls_last=True
            )

        present = set(report["query"].to_list())
        absent_rows = [
            {
                "query": q,
                "score": None,
                "fident": None,
                "evalue": None,
                "qcov": None,
                "tcov": None,
                "rev_strand": None,
                "hit_chrom": None,
                "hit_start": None,
                "hit_end": None,
                "mm10_accession": None,
                "mm10_class": None,
                "mm10_start": None,
                "mm10_end": None,
                "in_gold_standard": None,
                "rank": None,
            }
            for q in query_ids
            if q not in present
        ]
        if absent_rows:
            absent = pl.DataFrame(absent_rows, schema_overrides=report.schema)
            report = pl.concat([report, absent], how="diagonal").sort(
                ["query", "rank"], nulls_last=True
            )

        report.write_parquet(output.parquet)
        report.write_csv(output.tsv, separator="\t")

        # Aggregate stdout summary — scales to 300K+ queries without a
        # per-query line. Recall@k is the top-level metric from issue #120.
        n_queries = len(query_ids)
        queries_with_hits = report.filter(pl.col("rank").is_not_null())[
            "query"
        ].n_unique()
        queries_with_gold = sum(1 for q in query_ids if gold_by_query.get(q))
        recovered_at_k: dict[int, int] = {}
        for k in (1, 5, 10):
            q_with_gold_at_k = (
                report.filter(
                    pl.col("in_gold_standard")
                    & pl.col("rank").is_not_null()
                    & (pl.col("rank") <= k)
                )["query"]
                .unique()
                .to_list()
            )
            recovered_at_k[k] = len(set(q_with_gold_at_k) & set(query_ids))

        print(f"\n=== per-query report summary ({wildcards.aligner}) ===")
        print(f"  total queries (post-filter):      {n_queries:,}")
        print(f"  queries with ≥1 hit:              {queries_with_hits:,}")
        print(f"  queries with ≥1 gold partner:     {queries_with_gold:,}")
        for k in (1, 5, 10):
            denom = queries_with_gold
            recall = recovered_at_k[k] / denom if denom else float("nan")
            print(f"  recall@{k}: {recovered_at_k[k]:,}/{denom:,} = " f"{recall:.4f}")
        print(f"\nFull per-query report at {output.tsv}")

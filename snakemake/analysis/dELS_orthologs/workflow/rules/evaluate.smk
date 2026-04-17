"""Per-query report: for each hg38 query dELS, the top-k mmseqs2 hits in the
mm10 search window, annotated with the overlapping mm10 Registry-V4 cCRE
(any class) and gold-standard membership.

Replaces the older recall@k metric. The dELS-only filter applies to the hg38
query side only (per the tracking issue); on the mm10 side we keep cCREs of
every class so we can see hits to e.g. mouse `CA` cCREs that are the
orthologous element on the mouse side. See the issue body for the documented
hg38-dELS ↔ mm10-CA classification asymmetry at ZRS.
"""


rule per_query_report:
    input:
        hits="results/search/hits.tsv",
        query="results/cre/hg38/query.filtered.parquet",
        mm10_cres="results/cre/mm10/cres_window.parquet",
        gold="results/orthologs/hg38_mm10.tsv",
    output:
        parquet="results/eval/per_query_report.parquet",
        tsv="results/eval/per_query_report.tsv",
    params:
        top_k=config["report_top_k"],
    run:
        chrom, win_start, _ = get_search_window("mm10")

        hits = pl.read_csv(
            input.hits,
            separator="\t",
            has_header=False,
            new_columns=[
                "query",
                "target",
                "tstart",
                "tend",
                "bits",
                "evalue",
                "fident",
                "qcov",
                "tcov",
            ],
        )

        # mmseqs reports 1-based, end-inclusive positions; on reverse-strand
        # hits tstart > tend. Normalize to 0-based half-open BED coordinates
        # in absolute mm10 coordinates and record the strand.
        hits = hits.with_columns(
            (pl.col("tend") < pl.col("tstart")).alias("rev_strand"),
            (pl.min_horizontal("tstart", "tend") - 1 + win_start).alias("hit_start"),
            pl.max_horizontal("tstart", "tend").add(win_start).alias("hit_end"),
            pl.lit(chrom).alias("hit_chrom"),
        )

        mm10_cres = pl.read_parquet(input.mm10_cres)

        # Hit ↔ mm10 cCRE intersection (any class, any overlap > 0).
        if hits.height == 0:
            joined = pl.DataFrame(
                schema={
                    "query": pl.Utf8,
                    "bits": pl.Int64,
                    "evalue": pl.Float64,
                    "fident": pl.Float64,
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
                        "bits",
                        "evalue",
                        "fident",
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
        # Per-query set of gold-standard mm10 partners (for `in_gold_standard`
        # and to surface unrecovered gold partners).
        query_ids = pl.read_parquet(input.query)["accession"].to_list()

        gold_pairs = set(
            zip(
                gold["hg38_accession"].to_list(),
                gold["mm10_accession_gold"].to_list(),
            )
        )
        joined = joined.with_columns(
            pl.struct(["query", "mm10_accession"])
            .map_elements(
                lambda r: (r["query"], r["mm10_accession"]) in gold_pairs,
                return_dtype=pl.Boolean,
            )
            .alias("in_gold_standard"),
        )

        # Top-k hits per query by bits desc; one row per (query, hit). Hits
        # that fall in inter-cCRE space (no overlap) get null mm10 fields
        # but are still ranked.
        report = (
            joined.sort(["query", "bits"], descending=[False, True])
            .with_columns(
                pl.col("bits")
                .rank("ordinal", descending=True)
                .over("query")
                .cast(pl.Int64)
                .alias("rank"),
            )
            .filter(pl.col("rank") <= params.top_k)
            .sort(["query", "rank"])
        )

        # Surface gold partners that the search did NOT recover, as zero-bit
        # rows so the per-query view is honest about misses.
        recovered = set(
            zip(
                report["query"].to_list(),
                report.filter(pl.col("mm10_accession").is_not_null())[
                    "mm10_accession"
                ].to_list(),
            )
        ) | set(  # also count nulls so we don't double-add
            (q, m) for q, m in [] if False
        )
        missing_rows = []
        mm10_cre_lookup = {r["accession"]: r for r in mm10_cres.iter_rows(named=True)}
        for q in query_ids:
            for partner in {m for h, m in gold_pairs if h == q}:
                if (q, partner) not in recovered:
                    cre = mm10_cre_lookup.get(partner)
                    missing_rows.append(
                        {
                            "query": q,
                            "bits": None,
                            "evalue": None,
                            "fident": None,
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

            # Ensure every post-filter query has at least one row, so queries that
            # produced no mmseqs hits AND have no gold partner remain visible.
        present = set(report["query"].to_list())
        absent_rows = [
            {
                "query": q,
                "bits": None,
                "evalue": None,
                "fident": None,
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

        print("\n=== per-query report ===")
        # Pretty-print a compact summary for the run log.
        for q in query_ids:
            sub = report.filter(pl.col("query") == q)
            n_hits = sub.filter(pl.col("rank").is_not_null()).height
            n_gold_recovered = sub.filter(
                pl.col("in_gold_standard") & pl.col("rank").is_not_null()
            ).height
            n_gold_total = sum(1 for h, _m in gold_pairs if h == q)
            print(
                f"  {q}: {n_hits} mmseqs2 hits in window; "
                f"{n_gold_recovered}/{n_gold_total} gold-standard partner(s) recovered"
            )
        print(f"\nFull report at {output.tsv}")

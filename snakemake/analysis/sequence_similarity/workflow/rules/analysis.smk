"""Sanity check: search validation sequences against themselves.

Verifies the pipeline works correctly by searching val sequences against
themselves. Expected: adjacent sliding windows (50% overlap) produce hits,
and the hit count increases at lower coverage thresholds.

Only run for promoters — CDS uses the same search machinery so a single
interval type is sufficient.
"""


rule sanity_check_search:
    """Search validation sequences against themselves (sanity check)."""
    input:
        query_db="results/search/{dataset}/queryDB",
        query_db_type="results/search/{dataset}/queryDB.dbtype",
    output:
        result_index="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB.index",
        result_db_type="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB.dbtype",
    params:
        query_prefix="results/search/{dataset}/queryDB",
        result_prefix="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB",
        tmp_dir="results/sanity_check/{dataset}/tmp_id{identity}_cov{coverage}",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=lambda wildcards: float(wildcards.coverage),
        cov_mode=MMSEQS_COV_MODE,
    threads: workflow.cores
    resources:
        mem_mb=32000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {params.query_prefix} \
            {params.query_prefix} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --min-seq-id {params.identity} \
            -c {params.coverage} \
            --cov-mode {params.cov_mode} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule extract_sanity_check_hits:
    """Convert binary sanity check result DB to TSV."""
    input:
        query_db="results/search/{dataset}/queryDB",
        query_db_type="results/search/{dataset}/queryDB.dbtype",
        result_index="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB.index",
        result_db_type="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB.dbtype",
    output:
        tsv="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}.tsv",
    params:
        query_prefix="results/search/{dataset}/queryDB",
        result_prefix="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}/resultDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {params.query_prefix} \
            {params.query_prefix} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,fident,qcov,tcov"
        """


rule analyze_sanity_check:
    """Compute stats from sanity check search hits (including self-hits).

    For each validation sequence, count how many validation sequences
    are direct search hits. Self-hits are included as a sanity signal:
    every sequence should match itself, so val_matches >= 1 for all.
    """
    input:
        hits="results/sanity_check/{dataset}/hits_id{identity}_cov{coverage}.tsv",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/sanity_check/{dataset}/stats_id{identity}_cov{coverage}.parquet",
    run:
        hits = pl.read_csv(
            input.hits,
            separator="\t",
            has_header=False,
            new_columns=["query", "target", "fident", "qcov", "tcov"],
        )

        metadata = pl.read_parquet(input.metadata)
        val_ids = metadata.filter(pl.col("split") == "validation").select("id")
        total_val = val_ids.height

        # Count distinct val hits per val query (including self-hit)
        hits_per_query = (
            hits
            .group_by("query")
            .agg(pl.col("target").n_unique().alias("val_matches"))
        )

        val_matches = (
            val_ids
            .join(hits_per_query, left_on="id", right_on="query", how="left")
            .with_columns(pl.col("val_matches").fill_null(0))
        )

        match_counts = val_matches["val_matches"]
        seqs_with_matches = val_matches.filter(pl.col("val_matches") > 0).height
        pct_with_matches = 100.0 * seqs_with_matches / total_val if total_val > 0 else 0.0

        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "identity_threshold": [float(wildcards.identity)],
            "coverage_threshold": [float(wildcards.coverage)],
            "total_val_sequences": [total_val],
            "seqs_with_matches": [seqs_with_matches],
            "pct_with_matches": [pct_with_matches],
            "val_matches_mean": [float(match_counts.mean())],
            "val_matches_median": [float(match_counts.median())],
            "val_matches_max": [int(match_counts.max())],
        })

        stats.write_parquet(output.stats)

        print(f"\n=== Sanity Check: {wildcards.dataset} @ id={wildcards.identity} cov={wildcards.coverage} ===")
        print(f"  Total validation sequences: {total_val:,}")
        print(f"  Sequences with matches: {seqs_with_matches:,} ({pct_with_matches:.1f}%)")
        print(f"  Val matches per seq: median={stats['val_matches_median'][0]:.0f}, mean={stats['val_matches_mean'][0]:.1f}, max={stats['val_matches_max'][0]}")


rule aggregate_sanity_check:
    """Aggregate sanity check results across identity x coverage thresholds."""
    input:
        stats=expand(
            "results/sanity_check/{{dataset}}/stats_id{identity}_cov{coverage}.parquet",
            identity=get_identity_thresholds(),
            coverage=get_coverage_thresholds(),
        ),
    output:
        summary="results/sanity_check/{dataset}/summary.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs).sort(["coverage_threshold", "identity_threshold"])

        summary.write_parquet(output.summary)

        print(f"\n=== Sanity Check Summary: {wildcards.dataset} ===")
        print(summary.to_pandas().to_string(index=False))
        print("\nExpected behavior:")
        print("  - At lower coverage: more matches from overlapping sliding windows")
        print("  - With --strand 2: RC matches are found automatically")

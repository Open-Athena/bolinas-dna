GTEX_TISSUES = pd.read_csv("config/gtex_tissues.csv")
GTEX_DATASET_IDS = GTEX_TISSUES["dataset_id"].tolist()
_GTEX_TISSUE_BY_ID = dict(zip(GTEX_TISSUES["dataset_id"], GTEX_TISSUES["tissue_label"]))


# eQTL Catalogue r7 FTP URLs follow a fixed pattern keyed by (study, dataset).
# Verified for GTEx (QTS000015) — all 49 datasets have both the CS and the
# nominal-sumstats files under these paths. See:
# https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv
def _eqtl_cs_url(dataset_id):
    study = config["eqtl"]["catalogue_study_id"]
    return f"ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/{study}/{dataset_id}/{dataset_id}.credible_sets.tsv.gz"


def _eqtl_sumstats_url(dataset_id):
    study = config["eqtl"]["catalogue_study_id"]
    return f"ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/{study}/{dataset_id}/{dataset_id}.all.tsv.gz"


rule eqtl_parse_per_tissue:
    """Per-tissue: download credible_sets + nominal sumstats, parse, output parquet.

    Combines download + parse into one rule so the 3.5 GB nominal sumstats
    transits through a tempdir and is deleted on rule exit — no S3 round-trip
    for the raw TSV (would be 170 GB of waste across 49 tissues for files we
    only need to parse once).

    The parse: streams the nominal sumstats via `pl.scan_csv`, dedupes to one
    row per variant, left-joins per-tissue CS PIPs, 0-fills the PIP for
    tested-but-not-in-CS variants. The 0-fill is the load-bearing sentinel
    that makes downstream `label_variants_by_pip` produce a negative label
    for "tested but no signal" variants — see
    `bolinas.evals.catalogue_parser` docstring.

    Output: ~50–400 MB per tissue (depends on tissue sample size + cis-window
    variant density). Cached on S3 so subsequent rule firings are no-ops.
    """
    output:
        "results/eqtl/per_tissue/{dataset_id}.parquet",
    params:
        cs_url=lambda wc: _eqtl_cs_url(wc.dataset_id),
        sumstats_url=lambda wc: _eqtl_sumstats_url(wc.dataset_id),
    run:
        import subprocess
        import tempfile

        from bolinas.evals.catalogue_parser import (
            extract_tested_variants,
            merge_cs_and_sumstats,
            parse_credible_sets,
        )

        tissue_label = _GTEX_TISSUE_BY_ID[wildcards.dataset_id]
        with tempfile.TemporaryDirectory(prefix=f"eqtl_{wildcards.dataset_id}_") as tmp:
            cs_path = f"{tmp}/cs.tsv.gz"
            sumstats_path = f"{tmp}/sumstats.tsv.gz"
            # `--retry 5 --retry-delay 10` to ride out FTP hiccups; `--fail`
            # so curl exits nonzero on HTTP/FTP error (snakemake then re-runs).
            subprocess.run(
                ["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "10",
                 "-o", cs_path, params.cs_url],
                check=True,
            )
            subprocess.run(
                ["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "10",
                 "-o", sumstats_path, params.sumstats_url],
                check=True,
            )
            cs_df = parse_credible_sets(cs_path)
            sumstats_lf = extract_tested_variants(sumstats_path)
            merged = merge_cs_and_sumstats(cs_df, sumstats_lf, tissue_label)
            merged.sink_parquet(output[0])


rule eqtl_aggregate_tissues:
    """Cross-tissue per-variant labeling via `label_variants_by_pip`.

    See `src/bolinas/evals/labeling.py` for the cascade (positives if
    max(pip) > pip_pos, negatives if max(pip) < pip_neg, intermediate
    excluded by the otherwise→null→filter chain). The 0-fill from
    `merge_cs_and_sumstats` guarantees max(pip) is well-defined for every
    variant that appears in any tissue's sumstats — that's the change vs.
    Finucane that gives us the rich tested-but-no-signal negative pool.
    """
    input:
        expand(
            "results/eqtl/per_tissue/{dataset_id}.parquet",
            dataset_id=GTEX_DATASET_IDS,
        ),
    output:
        "results/eqtl/aggregated.parquet",
    run:
        pip_pos = config["eqtl"]["pip_pos_threshold"]
        pip_neg = config["eqtl"]["pip_neg_threshold"]
        # `use_null_pip_guard=False` because Catalogue SuSiE is single-method —
        # no SuSiE/FINEMAP-disagreement nulls to defend against (unlike
        # complex_traits). After `merge_cs_and_sumstats` 0-fills, the per-tissue
        # parquets contain no null pips anyway.
        labeled = label_variants_by_pip(
            pl.scan_parquet(list(input)),
            pip_pos_threshold=pip_pos,
            pip_neg_threshold=pip_neg,
            use_null_pip_guard=False,
            extra_aggs=[
                pl.col("maf").mean().alias("MAF"),
                pl.col("tissue")
                .filter(pl.col("pip") > pip_pos)
                .unique()
                .sort()
                .str.join(",")
                .alias("tissues"),
            ],
        )
        labeled.sort(COORDINATES).sink_parquet(output[0])


rule eqtl_annotate:
    input:
        "results/eqtl/aggregated.parquet",
        genome="results/genome.fa.gz",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/eqtl/annotated.parquet",
    run:
        genome = Genome(input.genome)
        V = (
            pl.read_parquet(input[0])
            .pipe(filter_chroms)
            .pipe(check_ref_alt, genome)
            .sort(COORDINATES)
        )
        # Per-chrom consequences attach via predicate pushdown (same trick as
        # complex_traits.smk:159-170): pos.is_in() lets parquet skip non-matching
        # row groups so each chrom's join only touches the relevant rows.
        results = []
        for path, chrom in zip(input.consequences, CHROMS):
            pos_chrom = V.filter(pl.col("chrom") == chrom)
            if pos_chrom.height == 0:
                continue
            cons_subset = (
                pl.scan_parquet(path)
                .filter(pl.col("pos").is_in(pos_chrom["pos"].unique().to_list()))
                .collect()
            )
            results.append(pos_chrom.join(cons_subset, on=COORDINATES, how="left"))
        pl.concat(results).write_parquet(output[0])


rule eqtl_dataset_all:
    input:
        "results/eqtl/annotated.parquet",
        "results/intervals/exon_pc.parquet",
        "results/intervals/exon_nc.parquet",
        "results/intervals/tss_pc.parquet",
        "results/intervals/tss_nc.parquet",
    output:
        "results/eqtl/dataset_all.parquet",
    run:
        build_dataset(
            pl.read_parquet(input[0]),
            exon_pc=pl.read_parquet(input[1]),
            exon_nc=pl.read_parquet(input[2]),
            tss_pc=pl.read_parquet(input[3]),
            tss_nc=pl.read_parquet(input[4]),
            exclude_consequences=config["exclude_consequences"],
            exon_proximal_dist=config["exon_proximal_dist"],
            tss_proximal_dist=config["tss_proximal_dist"],
            consequence_groups=config["consequence_groups"],
        ).write_parquet(output[0])


rule eqtl_dataset:
    input:
        "results/eqtl/dataset_all.parquet",
    output:
        "results/dataset_unsplit/eqtl.parquet",
    run:
        # Iter-33 locked design (issue #156). MAF scheme =
        # `MAF_TIERED_LOG8_DISTAL_ONLY` (tiered_v1 globally + local-log bins
        # for `distal`, which closes the asymptotic distal MAF residual leak
        # all global schemes left ★). NaN/null MAF dropped up-front; the
        # log_local window agg would otherwise propagate NaN.
        V = (
            pl.read_parquet(input[0])
            .filter(pl.col("MAF").is_finite() & pl.col("MAF").is_not_null())
            .pipe(add_subset_distance_bins)
        )
        V = add_tiered_maf_bin(
            V, MAF_TIERED_LOG8_DISTAL_ONLY, log_local_group_cols=CAT_BASE
        )
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                [
                    "distance_tss_pc",
                    "distance_tss_nc",
                    "distance_exon_pc",
                    "distance_exon_nc",
                    "MAF",
                ],
                CAT_BASE
                + [
                    "distance_tss_pc_bin",
                    "distance_tss_nc_bin",
                    "distance_exon_pc_bin",
                    "MAF_bin",
                ],
                k=1,
            )
            .with_columns(subset=pl.col("consequence_group"))
            .write_parquet(output[0])
        )

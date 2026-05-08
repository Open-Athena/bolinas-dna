FINEMAPPING_METHODS = ["SuSiE", "FINEMAP"]
FINEMAPPING_PIP_DIFF_THRESHOLD = 0.05
COMPLEX_TRAITS = pl.read_csv("config/complex_traits.csv")["trait"].to_list()


rule complex_traits_download_finemapping:
    output:
        "results/complex_traits/finemapping/{trait}/{method}.parquet",
    wildcard_constraints:
        method="|".join(FINEMAPPING_METHODS),
    params:
        url=lambda wc: f"https://huggingface.co/datasets/{config['complex_traits']['finemapping_repo']}/resolve/main/UKBB.{wc.trait}.{wc.method}.tsv.bgz",
    run:
        (
            pl.read_csv(
                params.url,
                separator="\t",
                null_values=["NA"],
                schema_overrides={"chromosome": pl.String},
                columns=[
                    "chromosome",
                    "position",
                    "allele1",
                    "allele2",
                    "rsid",
                    "pip",
                ],
            )
            .rename(
                {
                    "chromosome": "chrom",
                    "position": "pos",
                    "allele1": "ref",
                    "allele2": "alt",
                }
            )
            .pipe(filter_snp)
            .write_parquet(output[0])
        )


rule complex_traits_combine_methods:
    input:
        susie="results/complex_traits/finemapping/{trait}/SuSiE.parquet",
        finemap="results/complex_traits/finemapping/{trait}/FINEMAP.parquet",
    output:
        "results/complex_traits/finemapping/{trait}/combined.parquet",
    run:
        pip_defined_in_both_methods = (
            pl.col("pip_susie").is_not_null() & pl.col("pip_finemap").is_not_null()
        )
        (
            pl.read_parquet(input.susie)
            .join(
                pl.read_parquet(input.finemap),
                on=COORDINATES,
                how="full",
                suffix="_finemap",
            )
            .rename({"pip": "pip_susie", "rsid": "rsid_susie"})
            .with_columns(
                *(pl.coalesce(col, f"{col}_finemap").alias(col) for col in COORDINATES),
                pl.coalesce("rsid_susie", "rsid_finemap").alias("rsid"),
                pl.when(pip_defined_in_both_methods)
                .then(
                    pl.when(
                        (pl.col("pip_susie") - pl.col("pip_finemap")).abs()
                        <= FINEMAPPING_PIP_DIFF_THRESHOLD
                    )
                    .then((pl.col("pip_susie") + pl.col("pip_finemap")) / 2)
                    .otherwise(pl.lit(None))
                )
                .otherwise(pl.coalesce("pip_susie", "pip_finemap"))
                .alias("pip"),
            )
            .select([*COORDINATES, "rsid", "pip"])
            .write_parquet(output[0])
        )


rule complex_traits_aggregate_traits:
    input:
        expand(
            "results/complex_traits/finemapping/{trait}/combined.parquet",
            trait=COMPLEX_TRAITS,
        ),
    output:
        "results/complex_traits/finemapping/aggregated.parquet",
    run:
        high = config["complex_traits"]["pip_pos_threshold"]
        low = config["complex_traits"]["pip_neg_threshold"]
        any_null_pip = pl.col("pip").is_null().any()
        # Streaming: scan_parquet + sink_parquet so polars can chunk through the
        # 119-trait concat + group_by without materializing all ~50 GB of
        # per-trait fine-mapping data at once.
        (
            pl.concat(
                [
                    pl.scan_parquet(path).with_columns(trait=pl.lit(trait))
                    for path, trait in zip(input, COMPLEX_TRAITS)
                ]
            )
            .with_columns(
                pl.when(pl.col("pip") > high)
                .then(pl.col("trait"))
                .otherwise(pl.lit(None))
                .alias("trait")
            )
            .group_by(COORDINATES)
            .agg(
                pl.col("rsid").first(),
                pl.col("pip").max(),
                any_null_pip.alias("any_null_pip"),
                pl.col("trait").drop_nulls().unique(),
            )
            .with_columns(pl.col("trait").list.sort().list.join(",").alias("traits"))
            .with_columns(
                pl.when(pl.col("pip") > high)
                .then(pl.lit(True))
                .when((pl.col("pip") < low) & ~pl.col("any_null_pip"))
                .then(pl.lit(False))
                .otherwise(pl.lit(None))
                .alias("label")
            )
            .filter(pl.col("label").is_not_null())
            .drop(["trait", "any_null_pip"])
            .sort(COORDINATES)
            .sink_parquet(output[0])
        )


rule complex_traits_annotate:
    input:
        "results/complex_traits/finemapping/aggregated.parquet",
        "results/ldscore/UKBB.EUR.ldscore.parquet",
        genome="results/genome.fa.gz",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/complex_traits/annotated.parquet",
    run:
        ldscore = pl.read_parquet(input[1], columns=COORDINATES + ["MAF", "ld_score"])
        genome = Genome(input.genome)
        V = (
            pl.read_parquet(input[0])
            .join(ldscore, on=COORDINATES, how="left")
            # Drops high-PIP variants with very low MAF not in the LD-score file
            .filter(pl.col("ld_score").is_not_null())
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
            .pipe(filter_chroms)
            .pipe(check_ref_alt, genome)
            .sort(COORDINATES)
        )
        # Per-chrom consequences attach via predicate pushdown (Polars 1.40's
        # streaming left-join materializes the right side; pos.is_in() lets
        # the parquet reader skip non-matching row groups instead).
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


rule complex_traits_dataset_all:
    input:
        "results/complex_traits/annotated.parquet",
        "results/intervals/exon_pc.parquet",
        "results/intervals/exon_nc.parquet",
        "results/intervals/tss_pc.parquet",
        "results/intervals/tss_nc.parquet",
    output:
        "results/complex_traits/dataset_all.parquet",
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


rule complex_traits_dataset:
    input:
        "results/complex_traits/dataset_all.parquet",
    output:
        "results/dataset_unsplit/complex_traits.parquet",
    run:
        V = pl.read_parquet(input[0])
        # Drop the ~1.5% of rows where the gnomAD-MAF column is null/NaN — they
        # never reach a useful match group anyway (cdist on NaN distances ends
        # up at the back of argsort) and the upstream MAF binning needs finite
        # values. Iter-33 helpers do the same up-front filter.
        V = V.filter(pl.col("MAF").is_finite() & pl.col("MAF").is_not_null())
        # Iter-33 locked design (issue #156). Same distance bins as mendelian
        # (tss_pc + tss_nc on tss_proximal, exon_pc on splicing). MAF gets the
        # `MAF_TIERED_V1` per-subset scheme: 20bin for {distal, tss_proximal,
        # ncRNA}, 10bin for {3'UTR, 5'UTR, missense}, 5bin elsewhere — closes
        # every Bonf-significant MAF leak in the no-bin baseline while
        # recovering missense pairs vs. uniform 20bin (89 → 125). Combined
        # min-distance / closest-gene-id columns and ld_score are passthrough
        # metadata, not in the match key.
        V = V.with_columns(
            pl.when(pl.col("consequence_group") == "tss_proximal")
            .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_tss_pc_bin"),
            pl.when(pl.col("consequence_group") == "tss_proximal")
            .then(bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_tss_nc_bin"),
            pl.when(pl.col("consequence_group") == "splicing")
            .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_exon_pc_bin"),
        )
        V = add_tiered_maf_bin(V, MAF_TIERED_V1)
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
                [
                    "chrom",
                    "consequence_final",
                    "tss_closest_pc_gene_id",
                    "tss_closest_nc_gene_id",
                    "exon_closest_pc_gene_id",
                    "exon_closest_nc_gene_id",
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

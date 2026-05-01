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
        (
            pl.concat(
                [
                    pl.read_parquet(path).with_columns(trait=pl.lit(trait))
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
            .write_parquet(output[0])
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
        ldscore = pl.read_parquet(
            input[1], columns=COORDINATES + ["MAF", "ld_score"]
        )
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
        "results/intervals/exon.parquet",
        "results/intervals/tss.parquet",
    output:
        "results/complex_traits/dataset_all.parquet",
    run:
        build_dataset(
            pl.read_parquet(input[0]),
            pl.read_parquet(input[1]),
            pl.read_parquet(input[2]),
            config["exclude_consequences"],
            config["exon_proximal_dist"],
            config["tss_proximal_dist"],
            config["consequence_groups"],
        ).write_parquet(output[0])


rule complex_traits_dataset:
    input:
        "results/complex_traits/dataset_all.parquet",
    output:
        "results/dataset_unsplit/complex_traits.parquet",
    run:
        V = pl.read_parquet(input[0])
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                ["tss_dist", "exon_dist", "MAF", "ld_score"],
                [
                    "chrom",
                    "consequence_final",
                    "tss_closest_gene_id",
                    "exon_closest_gene_id",
                ],
                k=1,
            )
            .with_columns(subset=pl.col("consequence_group"))
            .write_parquet(output[0])
        )

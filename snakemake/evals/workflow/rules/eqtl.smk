rule eqtl_download:
    output:
        "results/eqtl/raw.tsv.bgz",
    params:
        url=config["eqtl"]["source_url"],
        billing_project=config["gcp_billing_project"],
    shell:
        "gsutil -u {params.billing_project} cp {params.url} {output}"


rule eqtl_aggregate:
    input:
        "results/eqtl/raw.tsv.bgz",
        "results/intervals/gene_biotype.parquet",
    output:
        "results/eqtl/aggregated.parquet",
    run:
        pip_pos = config["eqtl"]["pip_pos_threshold"]
        pip_neg = config["eqtl"]["pip_neg_threshold"]
        # Pre-filter to PIP extremes drops the middle-range bulk of rows up
        # front (a variant whose only tissue rows all have pip in [pip_neg,
        # pip_pos] would be excluded from labeling anyway). The streaming
        # scan_csv + sink_parquet keeps the ~5-15 GB decompressed input
        # from being materialized at once.
        # Also collect per-variant unique tissues, target gene IDs (no version),
        # and target gene biotype classes (collapsed to pc / nc).
        biotype_lf = (
            pl.scan_parquet(input[1])
            .with_columns(gene_stripped=pl.col("gene_id").str.split(".").list.get(0))
            .with_columns(
                biotype_class=pl.when(pl.col("gene_biotype") == "protein_coding")
                .then(pl.lit("pc"))
                .otherwise(pl.lit("nc"))
            )
            .select(["gene_stripped", "biotype_class"])
        )
        (
            pl.scan_csv(
                input[0],
                separator="\t",
                schema_overrides={
                    "chromosome": pl.String,
                    # `start`/`end` are unused (we parse hg38 coords from the
                    # variant_hg38 column instead, see below) but they're
                    # sometimes written in scientific notation (e.g. `4.8e+07`)
                    # so they need a Float schema to even parse.
                    "start": pl.Float64,
                    "end": pl.Float64,
                },
            )
            .filter((pl.col("pip") > pip_pos) | (pl.col("pip") < pip_neg))
            .with_columns(
                # The `chromosome`/`start`/`end` columns are NOT hg38 — they
                # appear to be the SuSiE input coordinates (often hg19).
                # `variant_hg38` is the actual hg38-lifted coordinate, format
                # `chr{chrom}_{pos}_{ref}_{alt}_b38`. Parse all four canonical
                # fields from it for safety.
                pl.col("variant_hg38")
                .str.strip_suffix("_b38")
                .str.split("_")
                .alias("_v"),
                gene_stripped=pl.col("gene").str.split(".").list.get(0),
            )
            .with_columns(
                pl.col("_v").list.get(0).str.strip_prefix("chr").alias("chrom"),
                pl.col("_v").list.get(1).cast(pl.Int64).alias("pos"),
                pl.col("_v").list.get(2).alias("ref"),
                pl.col("_v").list.get(3).alias("alt"),
            )
            .drop("_v")
            .join(biotype_lf, on="gene_stripped", how="left")
            .with_columns(pl.col("biotype_class").fill_null("nc"))
            .group_by(COORDINATES)
            .agg(
                pl.col("pip").max(),
                pl.col("maf").mean().alias("MAF"),
                pl.col("tissue")
                .filter(pl.col("pip") > pip_pos)
                .unique()
                .sort()
                .str.join(",")
                .alias("tissues"),
                pl.col("gene_stripped")
                .filter(pl.col("pip") > pip_pos)
                .unique()
                .sort()
                .str.join(",")
                .alias("genes"),
                pl.col("biotype_class")
                .filter(pl.col("pip") > pip_pos)
                .unique()
                .sort()
                .str.join(",")
                .alias("biotype_classes"),
            )
            .with_columns(
                pl.when(pl.col("pip") > pip_pos)
                .then(pl.lit(True))
                .when(pl.col("pip") < pip_neg)
                .then(pl.lit(False))
                .otherwise(pl.lit(None))
                .alias("label")
            )
            .filter(pl.col("label").is_not_null())
            .pipe(filter_snp)
            .sort(COORDINATES)
            .sink_parquet(output[0])
        )


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
        V = pl.read_parquet(input[0])
        # Per-biotype matching: same set as complex_traits + always-on MAF
        # features. TSS bins are active for both `tss_proximal` and `distal`
        # so distal pairs share the same 0-10kb sub-binning that fixed the
        # iter-26 distal leakage in complex_traits. Combined min-distance and
        # combined-closest-gene columns are passthrough metadata only — not
        # used in matching. tissues / genes / biotype_classes are also
        # passthrough.
        V = V.with_columns(
            pl.when(pl.col("consequence_group").is_in(["tss_proximal", "distal"]))
            .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_tss_pc_bin"),
            pl.when(pl.col("consequence_group").is_in(["tss_proximal", "distal"]))
            .then(bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_tss_nc_bin"),
            pl.when(pl.col("consequence_group") == "splicing")
            .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_exon_pc_bin"),
            pl.when(pl.col("consequence_group") == "splicing")
            .then(bin_feature("distance_exon_nc", EXON_DIST_BIN_EDGES))
            .otherwise(pl.lit(BIN_NA))
            .alias("distance_exon_nc_bin"),
            bin_feature("MAF", MAF_BIN_EDGES, right_closed=True).alias("MAF_bin"),
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
                    "distance_exon_nc_bin",
                    "MAF_bin",
                ],
                k=1,
            )
            .with_columns(subset=pl.col("consequence_group"))
            .write_parquet(output[0])
        )

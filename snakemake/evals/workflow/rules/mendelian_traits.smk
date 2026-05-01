rule mendelian_traits_positives:
    input:
        "results/omim/variants.parquet",
        "results/smedley/variants.parquet",
        "results/hgmd/variants.parquet",
        "results/gnomad/all.parquet",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/mendelian_traits/positives.parquet",
    run:
        # Concatenation order encodes priority: omim > smedley > hgmd
        positives = pl.concat(
            [
                pl.read_parquet(input[0]).with_columns(source=pl.lit("omim")),
                pl.read_parquet(input[1]).with_columns(source=pl.lit("smedley_et_al")),
                pl.read_parquet(input[2]).with_columns(source=pl.lit("hgmd")),
            ],
            how="diagonal_relaxed",
        ).unique(COORDINATES, keep="first", maintain_order=True)
        # Chrom-by-chrom AF lookup: bounds memory to one chrom of gnomAD
        # (~3 GB for chr1) instead of materializing the full ~700M-row file.
        gnomad_lf = pl.scan_parquet(input[3]).select(COORDINATES + ["AF"])
        af_pieces = []
        for chrom in positives["chrom"].unique().to_list():
            gnomad_chrom = gnomad_lf.filter(pl.col("chrom") == chrom).collect()
            pos_chrom = positives.filter(pl.col("chrom") == chrom).select(COORDINATES)
            af_pieces.append(pos_chrom.join(gnomad_chrom, on=COORDINATES, how="inner"))
        af_for_positives = pl.concat(af_pieces)
        V = (
            positives.join(af_for_positives, on=COORDINATES, how="left")
            # Variants not in gnomAD get AF = 0
            .with_columns(pl.col("AF").fill_null(0))
            .filter(pl.col("AF") < config["mendelian_traits"]["AF_threshold"])
            .sort(COORDINATES)
        )
        results = []
        for path, chrom in zip(input.consequences, CHROMS):
            chrom_variants = V.filter(pl.col("chrom") == chrom).lazy()
            consequences_lf = pl.scan_parquet(path)
            joined = chrom_variants.join(
                consequences_lf,
                on=COORDINATES,
                how="left",
                maintain_order="left",
            ).collect(engine="streaming")
            results.append(joined)
        pl.concat(results).write_parquet(output[0])


rule mendelian_traits_dataset_all:
    input:
        "results/mendelian_traits/positives.parquet",
        "results/gnomad/common.parquet",
        "results/intervals/exon.parquet",
        "results/intervals/tss.parquet",
    output:
        "results/mendelian_traits/dataset_all.parquet",
    run:
        V = pl.concat(
            [
                pl.read_parquet(input[0]).with_columns(label=pl.lit(True)),
                pl.read_parquet(input[1]).with_columns(label=pl.lit(False)),
            ],
            how="diagonal_relaxed",
        )
        build_dataset(
            V,
            pl.read_parquet(input[2]),
            pl.read_parquet(input[3]),
            config["exclude_consequences"],
            config["exon_proximal_dist"],
            config["tss_proximal_dist"],
            config["consequence_groups"],
        ).write_parquet(output[0])


rule mendelian_traits_dataset:
    input:
        "results/mendelian_traits/dataset_all.parquet",
    output:
        "results/dataset_unsplit/mendelian_traits.parquet",
    run:
        V = pl.read_parquet(input[0])
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                ["tss_dist", "exon_dist"],
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

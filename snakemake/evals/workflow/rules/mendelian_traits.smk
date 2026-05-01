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
        import pyarrow.compute as pc
        import pyarrow.dataset as pads

        # Concatenation order encodes priority: omim > smedley > hgmd
        positives = pl.concat(
            [
                pl.read_parquet(input[0]).with_columns(source=pl.lit("omim")),
                pl.read_parquet(input[1]).with_columns(source=pl.lit("smedley_et_al")),
                pl.read_parquet(input[2]).with_columns(source=pl.lit("hgmd")),
            ],
            how="diagonal_relaxed",
        ).unique(COORDINATES, keep="first", maintain_order=True)
        # AF lookup via pyarrow.dataset: predicate pushdown on (chrom, pos)
        # skips parquet row groups whose ranges don't intersect the few
        # thousand positive coordinates, so peak memory is bounded by the
        # matched rows + a few row groups, not the full ~700M-row gnomAD.
        # (Polars 1.40's streaming engine quietly fell back to eager reads
        # for this query — pyarrow's predicate pushdown is more reliable.)
        gnomad_ds = pads.dataset(input[3], format="parquet")
        af_pieces = []
        for chrom, pos_chrom in positives.group_by("chrom"):
            chrom_value = chrom[0] if isinstance(chrom, tuple) else chrom
            filter_expr = (pc.field("chrom") == chrom_value) & pc.field("pos").isin(
                pos_chrom["pos"].to_list()
            )
            table = gnomad_ds.to_table(
                columns=COORDINATES + ["AF"],
                filter=filter_expr,
            )
            if table.num_rows == 0:
                continue
            gnomad_chunk = pl.from_arrow(table)
            af_pieces.append(
                pos_chrom.select(COORDINATES).join(
                    gnomad_chunk, on=COORDINATES, how="inner"
                )
            )
        af_for_positives = (
            pl.concat(af_pieces)
            if af_pieces
            else pl.DataFrame(
                schema={
                    "chrom": pl.Utf8,
                    "pos": pl.Int64,
                    "ref": pl.Utf8,
                    "alt": pl.Utf8,
                    "AF": pl.Float64,
                }
            )
        )
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

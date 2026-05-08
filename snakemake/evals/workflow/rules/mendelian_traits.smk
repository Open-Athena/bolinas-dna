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
        # AF lookup: scan + filter by pos.is_in. The parquet is sorted by
        # (chrom, pos) with row-group min/max stats, so this predicate is
        # pushed down — only row groups whose pos range overlaps the few
        # thousand positive positions are read. ~5 s, ~1 GB peak vs. the
        # ~60 GB OOM from a naive read_parquet + join (which materializes
        # the full ~700M-row AF column).
        candidates = (
            pl.scan_parquet(input[3])
            .select(COORDINATES + ["AF"])
            .filter(pl.col("pos").is_in(positives["pos"].unique().to_list()))
            .collect()
        )
        af_for_positives = positives.select(COORDINATES).join(
            candidates, on=COORDINATES, how="inner"
        )
        V = (
            positives.join(af_for_positives, on=COORDINATES, how="left")
            # Variants not in gnomAD get AF = 0
            .with_columns(pl.col("AF").fill_null(0))
            .filter(pl.col("AF") < config["mendelian_traits"]["AF_threshold"])
            .sort(COORDINATES)
        )
        # Per-chrom consequences attach via predicate pushdown (same trick as
        # the AF lookup above). Polars 1.40's streaming left-join materializes
        # the right side; pos.is_in() lets parquet skip non-matching row groups
        # so each chrom's join touches only a few thousand rows.
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


rule mendelian_traits_dataset_all:
    input:
        "results/mendelian_traits/positives.parquet",
        "results/gnomad/common.parquet",
        "results/intervals/exon_pc.parquet",
        "results/intervals/exon_nc.parquet",
        "results/intervals/tss_pc.parquet",
        "results/intervals/tss_nc.parquet",
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
            exon_pc=pl.read_parquet(input[2]),
            exon_nc=pl.read_parquet(input[3]),
            tss_pc=pl.read_parquet(input[4]),
            tss_nc=pl.read_parquet(input[5]),
            exclude_consequences=config["exclude_consequences"],
            exon_proximal_dist=config["exon_proximal_dist"],
            tss_proximal_dist=config["tss_proximal_dist"],
            consequence_groups=config["consequence_groups"],
        ).write_parquet(output[0])


rule mendelian_traits_dataset:
    input:
        "results/mendelian_traits/dataset_all.parquet",
    output:
        "results/dataset_unsplit/mendelian_traits.parquet",
    run:
        # Iter-33 locked design (issue #156). mendelian has no MAF column,
        # so no MAF_bin.
        V = add_subset_distance_bins(pl.read_parquet(input[0]))
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                [
                    "distance_tss_pc",
                    "distance_tss_nc",
                    "distance_exon_pc",
                    "distance_exon_nc",
                ],
                CAT_BASE
                + [
                    "distance_tss_pc_bin",
                    "distance_tss_nc_bin",
                    "distance_exon_pc_bin",
                ],
                k=1,
            )
            .with_columns(subset=pl.col("consequence_group"))
            .write_parquet(output[0])
        )

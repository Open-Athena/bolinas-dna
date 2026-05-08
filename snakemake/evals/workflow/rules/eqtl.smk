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
        # Per-variant labeling delegated to `label_variants_by_pip` (in
        # `src/bolinas/evals/labeling.py`, fully unit-tested in
        # `tests/evals/test_labeling.py`). Notes:
        #   - Labels come from `max(pip)` across the tissues that
        #     fine-mapped this variant. Intermediate `max(pip)` (in
        #     `[pip_neg, pip_pos]`) is excluded by the cascade's
        #     `otherwise(None)` + `filter(label.is_not_null())`. Critically,
        #     do NOT row-filter to extreme PIPs before the helper — that
        #     would silently mislabel a variant with tissue PIPs
        #     `[0.001, 0.5]` as a clean negative (`max = 0.001 < pip_neg`)
        #     when its true `max = 0.5` is intermediate and should
        #     exclude the variant. (This was the iter-pre-fix bug.)
        #   - `use_null_pip_guard=False`: source is SuSiE-only, no
        #     SuSiE/FINEMAP combine step that would emit null PIPs on
        #     disagreement. There's no quality-flag to guard against.
        #   - A variant only has rows for the tissues that fine-mapped
        #     it. Negatives are NOT required to be tested in all 49
        #     tissues — typical fine-mapping outputs only cover
        #     significant regions.
        # Per-variant unique tissues, target gene IDs (no version), and
        # target gene biotype classes (pc / nc) are collected via
        # `extra_aggs`, filtered to tissues where the variant was a
        # positive.
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
        rows = (
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
        )
        labeled = label_variants_by_pip(
            rows,
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
            ],
        )
        (
            labeled.pipe(filter_snp)
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

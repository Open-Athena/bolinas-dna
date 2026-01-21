# TODO: this dataset is deprecated as it misses a small proportion of  variants
# (e.g. multi-SNV)
# should ideally reprocess gnomAD from scratch


rule gnomad_download:
    output:
        "results/gnomad.parquet",
    run:
        (
            pl.read_parquet(
                "hf://datasets/songlab/deprecated-full-gnomad/test.parquet",
                columns=COORDS + ["AF"],
            )
            .rename({"AF": "label"})
            .write_parquet(output[0])
        )


rule gnomad_filter_pls:
    input:
        "results/gnomad.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/gnomad_pls.parquet",
    run:
        V = pd.read_parquet(input[0])
        PLS = pd.read_parquet(input[1]).query("cre_class == 'PLS'")

        V_filtered_list = []

        for chrom in CHROMS:
            V_chrom = V[V.chrom == chrom].copy()

            if len(V_chrom) == 0:
                continue

            PLS_chrom = PLS[PLS.chrom == chrom]

            if len(PLS_chrom) == 0:
                continue

            V_chrom["start"] = V_chrom.pos - 1
            V_chrom["end"] = V_chrom.pos

            V_chrom = bf.coverage(V_chrom, PLS_chrom)
            V_chrom = V_chrom[V_chrom.coverage > 0]
            V_chrom = V_chrom.drop(columns=["start", "end", "coverage"])

            V_filtered_list.append(V_chrom)

        V = pd.concat(V_filtered_list, ignore_index=True)
        V.to_parquet(output[0], index=False)


rule gnomad_pls_dataset_v1:
    input:
        "results/gnomad_pls.parquet",
    output:
        expand(
            "results/dataset/gnomad_pls_v1/{split}.parquet",
            split=SPLITS,
        ),
    run:
        n_bins = 100
        max_per_bin = 15_000 // n_bins

        V = pd.read_parquet(input[0])
        V["af_bin"] = (V["label"] * n_bins).astype(int).clip(0, n_bins - 1)

        for split, path in zip(SPLITS, output):
            V_split = V[V.chrom.isin(SPLIT_CHROMS[split])]
            samples = []
            for bin_id in range(n_bins):
                bin_data = V_split[V_split.af_bin == bin_id]
                n_sample = min(len(bin_data), max_per_bin)
                samples.append(bin_data.sample(n=n_sample, random_state=42))
            (
                pd.concat(samples, ignore_index=True)
                .drop(columns=["af_bin"])
                .sort_values(COORDS)
                .to_parquet(path, index=False)
            )


rule gnomad_v2_download:
    output:
        "results/gnomad_v2.parquet",
    run:
        (
            pl.read_parquet(
                "hf://datasets/songlab/gnomad/test.parquet",
                columns=COORDS + ["label", "consequence"],
            ).write_parquet(output[0])
        )


rule gnomad_v2_filter_pls:
    input:
        "results/gnomad_v2.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/gnomad_v2_pls.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V.query("consequence == 'upstream_gene'")
        print(V)
        PLS = pd.read_parquet(input[1]).query("cre_class == 'PLS'")

        V_filtered_list = []

        for chrom in CHROMS:
            V_chrom = V[V.chrom == chrom].copy()

            if len(V_chrom) == 0:
                continue

            PLS_chrom = PLS[PLS.chrom == chrom]

            if len(PLS_chrom) == 0:
                continue

            V_chrom["start"] = V_chrom.pos - 1
            V_chrom["end"] = V_chrom.pos

            V_chrom = bf.coverage(V_chrom, PLS_chrom)
            V_chrom = V_chrom[V_chrom.coverage > 0]
            V_chrom = V_chrom.drop(columns=["start", "end", "coverage"])

            V_filtered_list.append(V_chrom)

        V = pd.concat(V_filtered_list, ignore_index=True)
        V.to_parquet(output[0], index=False)


rule gnomad_v2_pls_dataset:
    input:
        "results/gnomad_v2_pls.parquet",
    output:
        expand(
            "results/dataset/gnomad_pls_v2/{split}.parquet",
            split=SPLITS,
        ),
    run:
        n_per_label = 5_000

        V = pd.read_parquet(input[0])

        for split, path in zip(SPLITS, output):
            V_split = V[V.chrom.isin(SPLIT_CHROMS[split])]
            samples = []
            for label_value in V_split["label"].unique():
                label_data = V_split[V_split.label == label_value]
                n_sample = min(len(label_data), n_per_label)
                samples.append(label_data.sample(n=n_sample, random_state=42))
            (
                pd.concat(samples, ignore_index=True)
                .sort_values(COORDS)
                .to_parquet(path, index=False)
            )

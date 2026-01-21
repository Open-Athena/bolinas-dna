rule gwas_coding_dataset:
    output:
        expand(
            "results/dataset/gwas_coding/{split}.parquet",
            split=SPLITS,
        ),
    run:
        V = pd.read_parquet(f"hf://datasets/songlab/ukb_finemapped_coding/test.parquet")
        for split, path in zip(SPLITS, output):
            V[V.chrom.isin(SPLIT_CHROMS[split])].to_parquet(path, index=False)

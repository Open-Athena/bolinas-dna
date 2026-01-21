rule clinvar_missense_dataset:
    output:
        expand(
            "results/dataset/clinvar_missense/{split}.parquet",
            split=SPLITS,
        ),
    run:
        V = pd.read_parquet(f"hf://datasets/songlab/clinvar_vs_benign/test.parquet")
        V.label = V.label == "Pathogenic"
        n_samples_per_label = 5000
        for split, path in zip(SPLITS, output):
            V_split = V[V.chrom.isin(SPLIT_CHROMS[split])]
            V_split = (
                V_split.groupby("label")
                .sample(n=n_samples_per_label, random_state=42)
                .reset_index(drop=True)
            )
            V_split = V_split.sort_values(COORDS)
            V_split.to_parquet(path, index=False)

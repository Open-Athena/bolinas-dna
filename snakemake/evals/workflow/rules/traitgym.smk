rule traitgym_dataset:
    output:
        expand(
            "results/dataset/traitgym_{{traits}}/{split}.parquet",
            split=SPLITS,
        ),
    run:
        traits = wildcards.traits
        V = pd.read_parquet(
            f"hf://datasets/songlab/TraitGym/{traits}_traits_matched_9/test.parquet"
        )
        V = V.set_index(COORDS)
        subsets = [
            "non_coding_transcript_exon_variant",
            "3_prime_UTR_variant",
            "5_prime_UTR_variant",
            "nonexonic_AND_proximal",
            "nonexonic_AND_distal",
        ]
        for subset in subsets:
            V_subset = pd.read_parquet(
                f"hf://datasets/songlab/TraitGym/{traits}_traits_matched_9/subset/{subset}.parquet"
            )
            V_subset = V_subset.set_index(COORDS)
            V.loc[V_subset.index, "subset"] = subset
        V = V.reset_index()
        for split, path in zip(SPLITS, output):
            V[V.chrom.isin(SPLIT_CHROMS[split])].to_parquet(path, index=False)

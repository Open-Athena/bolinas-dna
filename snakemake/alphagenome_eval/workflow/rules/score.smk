"""AlphaGenome scoring + per-track aggregation.

Two rules: the expensive API call (`compute_per_track_l2`) keeps its full
per-track output on S3 so the aggregation protocol can change later without
re-spending the API budget. The cheap `aggregate_max` rule reads that and
produces the slim `scores/{dataset}.parquet` consumed by `compute_metrics`.
"""


rule compute_per_track_l2:
    """Forward-strand AlphaGenome score per variant, one column per track."""
    output:
        "results/per_track_l2/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    threads: NUM_WORKERS
    run:
        assert os.environ.get(
            "ALPHA_GENOME_API_KEY"
        ), "ALPHA_GENOME_API_KEY env var not set"

        hf_path = f"{INPUT_HF_PREFIX}_{wildcards.dataset}"
        ds = load_dataset(hf_path, split=SPLIT).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in ds.columns, f"dataset missing column {col!r}"

        if SUBSET_N_PAIRS is not None:
            keep = ds["match_group"].drop_duplicates().head(int(SUBSET_N_PAIRS))
            ds = ds[ds["match_group"].isin(keep)].reset_index(drop=True)
            print(
                f"[alphagenome_eval] {wildcards.dataset}: "
                f"subset_n_pairs={SUBSET_N_PAIRS} → {len(ds)} variants"
            )

        per_track = score_variants_alphagenome(
            ds[["chrom", "pos", "ref", "alt"]],
            num_workers=NUM_WORKERS,
        )
        assert len(per_track) == len(ds)

        out = pd.concat(
            [
                ds[list(REQUIRED_VARIANT_COLUMNS)].reset_index(drop=True),
                per_track.reset_index(drop=True),
            ],
            axis=1,
        )
        out.to_parquet(output[0], index=False)
        n_track_cols = len(per_track.columns)
        print(
            f"[alphagenome_eval] {wildcards.dataset} ({SPLIT}): "
            f"n={len(out)} tracks={n_track_cols}"
        )


rule aggregate_max:
    """Collapse per-track L2 scores to a single max-across-tracks column."""
    input:
        "results/per_track_l2/{dataset}.parquet",
    output:
        "results/scores/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        df = pd.read_parquet(input[0])
        track_cols = [c for c in df.columns if c not in REQUIRED_VARIANT_COLUMNS]
        assert track_cols, "no per-track columns found in input parquet"

        out = df[list(REQUIRED_VARIANT_COLUMNS)].copy()
        out[SCORE_COLUMN] = df[track_cols].max(axis=1)
        assert (
            out[SCORE_COLUMN].notna().all()
        ), f"NaN in {SCORE_COLUMN} after max-across-tracks aggregation"
        out.to_parquet(output[0], index=False)
        print(
            f"[alphagenome_eval] {wildcards.dataset}: max-aggregated "
            f"{len(track_cols)} tracks → '{SCORE_COLUMN}' "
            f"min={out[SCORE_COLUMN].min():.3f} "
            f"max={out[SCORE_COLUMN].max():.3f}"
        )

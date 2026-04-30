rule score_variants:
    input:
        bw="results/conservation/{score}.bw",
    output:
        "results/conservation_traitgym_v2/{score}_{split}.parquet",
    wildcard_constraints:
        score="|".join(CONSERVATION_TRACKS),
        split="|".join(SPLITS),
    run:
        ds = load_dataset(DATASET_HF_PATH, split=wildcards.split).to_pandas()
        for col in ("chrom", "pos", "ref", "alt", "label", "subset"):
            assert col in ds.columns, f"dataset missing column {col!r}"

        scores = score_variants_at_positions(ds, input.bw)
        assert len(scores) == len(ds)

        out = ds[["chrom", "pos", "ref", "alt", "label", "subset"]].copy()
        out["score"] = scores
        n_nan = int(out["score"].isna().sum())
        print(
            f"[conservation_eval] {wildcards.score} {wildcards.split}: "
            f"n={len(out)} n_nan={n_nan} ({100* n_nan/ len(out):.2f}%) "
            f"score_min={out['score'].min():.3f} max={out['score'].max():.3f}"
        )
        out.to_parquet(output[0], index=False)


rule aggregate_metrics:
    input:
        parquets=lambda wc: expand(
            "results/conservation_traitgym_v2/{score}_{{split}}.parquet",
            score=SCORES,
        ),
    output:
        metrics="results/conservation_traitgym_v2/metrics_{split}.parquet",
        markdown="results/conservation_traitgym_v2/results_table_{split}.md",
    wildcard_constraints:
        split="|".join(SPLITS),
    run:
        from bolinas.evals.conservation import aggregate_traitgym_metrics
        from pathlib import Path

        parquet_paths = {
            score: f"results/conservation_traitgym_v2/{score}_{wildcards.split}.parquet"
            for score in SCORES
        }
        metrics, md = aggregate_traitgym_metrics(parquet_paths)
        metrics["split"] = wildcards.split
        metrics.to_parquet(output.metrics, index=False)
        Path(output.markdown).write_text(md)
        print(f"[conservation_eval] wrote {output.markdown}")
        print(md)

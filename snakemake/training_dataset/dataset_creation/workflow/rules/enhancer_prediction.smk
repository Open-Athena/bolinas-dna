ENHANCER_CONFIG = config.get("enhancer_prediction", {})


# Prevent {g} from matching path separators in enhancer prediction rules
wildcard_constraints:
    g="[^/]+",


rule extract_exons:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/exons/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        exons = get_exons_for_masking(ann)
        exons.write_parquet(output[0])


rule scannable_regions:
    input:
        defined="results/intervals/defined/{g}.bed.gz",
        exons="results/intervals/exons/{g}.parquet",
    output:
        "results/intervals/scannable/{g}.bed.gz",
    run:
        defined = GenomicSet.read_bed(input.defined)
        exons = GenomicSet.read_parquet(input.exons)
        scannable = defined - exons
        scannable.write_bed(output[0])


rule enhancer_prediction_windows:
    input:
        "results/intervals/scannable/{g}.bed.gz",
    output:
        "results/enhancer_predictions/windows/{g}.parquet",
    run:
        window_size = ENHANCER_CONFIG["window_size"]
        step_size = ENHANCER_CONFIG["step_size"]

        regions = pd.read_csv(
            input[0], sep="\t", header=None, names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )

        chroms, starts, ends = [], [], []
        for _, row in regions.iterrows():
            for w_start, w_end in sliding_windows(
                row["start"], row["end"], window_size, step_size
            ):
                chroms.append(row["chrom"])
                starts.append(w_start)
                ends.append(w_end)

        windows = pl.DataFrame({
            "chrom": chroms,
            "start": starts,
            "end": ends,
        })
        windows.write_parquet(output[0])


rule predict_enhancers:
    input:
        genome="results/genome/{g}.2bit",
        windows="results/enhancer_predictions/windows/{g}.parquet",
        checkpoint=storage(ENHANCER_CONFIG["checkpoint"]),
    output:
        "results/enhancer_predictions/{g}.parquet",
    params:
        batch_size=ENHANCER_CONFIG.get("batch_size", 512),
        num_workers=ENHANCER_CONFIG.get("num_workers", 4),
    threads: workflow.cores
    shell:
        """
        uv run python -m bolinas.enhancer_classification.predict_genome \
            --genome {input.genome} \
            --checkpoint {input.checkpoint} \
            --windows {input.windows} \
            --batch-size {params.batch_size} \
            --num-workers {params.num_workers} \
            --output {output}
        """


rule intervals_recipe_v19:
    input:
        "results/enhancer_predictions/{g}.parquet",
    output:
        "results/intervals/recipe/v19/{g}.bed.gz",
    run:
        threshold = ENHANCER_CONFIG["threshold"]
        df = pl.read_parquet(input[0])
        df = df.filter(pl.col("logit") >= threshold)
        intervals = GenomicSet(df.select(["chrom", "start", "end"]))
        intervals.write_bed(output[0])

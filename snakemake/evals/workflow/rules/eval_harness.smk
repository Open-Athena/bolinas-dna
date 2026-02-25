ruleorder: materialize_eval_harness_dataset > traitgym_dataset


rule download_genome:
    output:
        "results/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget {params.url} -O {output}"


rule materialize_eval_harness_dataset:
    input:
        parquet="results/dataset/{dataset}/{split}.parquet",
        genome="results/genome.fa.gz",
    output:
        "results/dataset/{dataset}_harness_{window_size}/{split}.parquet",
    run:
        genome = Genome(input.genome)
        ds = Dataset.from_parquet(input.parquet)
        ds = materialize_sequences(ds, genome, int(wildcards.window_size))
        ds.to_parquet(output[0])

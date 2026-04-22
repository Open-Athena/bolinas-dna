"""Sanity check: hg38 dELS (post-filter) searched against themselves.

Pinned to flank=0: self-alignment is flank-invariant by construction (query == target),
so running it across the flank sweep would produce essentially identical results.
"""


rule sanity_self_search:
    input:
        query_db="results/search/flank_0/queryDB",
        query_dbtype="results/search/flank_0/queryDB.dbtype",
    output:
        result_index="results/sanity/selfDB.index",
        result_dbtype="results/sanity/selfDB.dbtype",
    params:
        result_prefix="results/sanity/selfDB",
        tmp_dir="results/sanity/tmp",
        sensitivity=config["mmseqs2"]["sensitivity"],
        max_accept=config["mmseqs2"]["max_accept"],
        split_memory_limit=config["mmseqs2"].get("split_memory_limit", "12G"),
    threads: workflow.cores
    resources:
        mem_mb=config["mmseqs2"].get("mem_mb", 14000),
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {input.query_db} \
            {input.query_db} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --split-memory-limit {params.split_memory_limit} \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule sanity_self_convertalis:
    input:
        query_db="results/search/flank_0/queryDB",
        result_index="results/sanity/selfDB.index",
    output:
        tsv="results/sanity/self_hits.tsv",
    params:
        result_prefix="results/sanity/selfDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {input.query_db} \
            {input.query_db} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,bits,fident"
        """


rule sanity_self_recall:
    input:
        hits="results/sanity/self_hits.tsv",
        hg38_dels="results/cre/hg38/query.filtered.parquet",
    output:
        "results/sanity/self_recall.parquet",
    run:
        hits = pl.read_csv(
            input.hits,
            separator="\t",
            has_header=False,
            new_columns=["query", "target", "bits", "fident"],
        )
        dels = pl.read_parquet(input.hg38_dels)
        n_queries = dels.height
        all_query_ids = set(dels["accession"].to_list())

        top1 = (
            hits.sort(["query", "bits"], descending=[False, True])
            .group_by("query", maintain_order=True)
            .first()
        )
        n_self = top1.filter(pl.col("query") == pl.col("target")).height
        n_wrong_top1 = top1.filter(pl.col("query") != pl.col("target")).height
        n_no_hits = len(all_query_ids - set(hits["query"].to_list()))
        recall_at_1 = n_self / n_queries if n_queries else float("nan")

        out = pl.DataFrame(
            {
                "n_queries": [n_queries],
                "n_self_top1": [n_self],
                "n_wrong_top1": [n_wrong_top1],
                "n_no_hits": [n_no_hits],
                "recall_at_1": [recall_at_1],
            }
        )
        out.write_parquet(output[0])
        print("\n=== sanity self-alignment (post-filter) ===")
        print(out.to_pandas().to_string(index=False))
        if n_wrong_top1 > 0:
            print(
                f"WARNING: {n_wrong_top1} of {n_queries} dELS had a non-self top-1 hit "
                f"— this would indicate a bug in the search machinery."
            )
        if n_no_hits > 0:
            print(
                f"NOTE: {n_no_hits} of {n_queries} (post-filter) dELS produced no self-hit. "
                f"Consider lowering `max_soft_masked_frac`."
            )

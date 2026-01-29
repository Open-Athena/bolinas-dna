rule chrom_sizes:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/chrom_sizes/{g}.tsv",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


# just adds the start=0 column, but in human it also filters to standard chroms
rule extract_all:
    input:
        "results/chrom_sizes/{g}.tsv",
    output:
        "results/intervals/all/{g}.bed.gz",
    run:
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "end"])
        df["start"] = 0
        # we want to filter to chromosomes, and exclude unplaced scaffolds,
        # alt. haplotypes, etc.
        # unfortunately no way to filter chroms based on prefix, AFAIK
        # in human we want to make sure to keep the standard chroms
        if wildcards.g == "GCF_000001405.40":
            df = df[df.chrom.str[:2] == "NC"]
        df = df[["chrom", "start", "end"]]
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule extract_undefined:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/intervals/undefined/{g}.bed.gz",
    params:
        "results/intervals/undefined/{g}.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {params} -nBed && gzip {params}"


rule extract_defined:
    input:
        "results/intervals/all/{g}.bed.gz",
        "results/intervals/undefined/{g}.bed.gz",
    output:
        "results/intervals/defined/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "bedtools subtract -a {input[0]} -b {input[1]} | gzip > {output}"


rule make_windows:
    input:
        "results/intervals/{intervals}/{g}.bed.gz",
    output:
        "results/intervals/windows/{intervals}/{w}/{s}/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        bedtools makewindows -b {input[0]} -w {wildcards.w} -s {wildcards.s} | \
        awk '$3-$2 == {wildcards.w}' | \
        gzip > {output}
        """


# promoters from protein-coding transcripts, similar to gpn-animal-promoter-dataset
# described in TraitGym paper
rule intervals_recipe_v1:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v1/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256

        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=promoter_n_upstream,
            n_downstream=promoter_n_downstream,
            mRNA_only=True,
            within_bounds=defined,
        )
        promoters.write_bed(output[0])


rule intervals_recipe_v4:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v4/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256

        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=promoter_n_upstream,
            n_downstream=promoter_n_downstream,
            mRNA_only=False,
            within_bounds=defined,
        )
        promoters.write_bed(output[0])


# CDS regions only
rule intervals_recipe_v3:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v3/{g}.bed.gz",
    run:
        min_size = 512

        ann = load_annotation(input[0])
        cds = get_cds(ann)
        defined = GenomicSet(read_bed_to_pandas(input[1]))
        intervals = cds.expand_min_size(min_size)
        intervals = intervals & defined
        write_pandas_to_bed(intervals.to_pandas(), output[0])


rule extract_cds:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/cds/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        cds = get_cds(ann)
        assert cds.n_intervals() > 0, f"No CDS regions found for {wildcards.g}"
        cds.write_parquet(output[0])


rule extract_5_prime_utr:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/5_prime_utr/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        utr = get_5_prime_utr(ann)
        # Allow empty UTR sets for genomes without UTR annotations
        utr.write_parquet(output[0])


rule extract_3_prime_utr:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/3_prime_utr/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        utr = get_3_prime_utr(ann)
        # Allow empty UTR sets for genomes without UTR annotations
        utr.write_parquet(output[0])


rule extract_promoters:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/all/{g}.bed.gz",
    output:
        "results/intervals/promoters/{upstream}/{downstream}/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        bounds = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=int(wildcards.upstream),
            n_downstream=int(wildcards.downstream),
            mRNA_only=False,
            within_bounds=bounds,
        )
        assert (
            promoters.n_intervals() > 0
        ), f"No promoter regions found for {wildcards.g}"
        promoters.write_parquet(output[0])


rule extract_ncrna_exons:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/ncrna_exons/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        ncrna = get_ncrna_exons(ann)
        # Allow empty ncRNA sets for genomes without ncRNA annotations
        ncrna.write_parquet(output[0])


rule parquet_to_bed:
    input:
        "results/intervals/{region}/{g}.parquet",
    output:
        "results/bed/{g}/{region}.bed",
    run:
        GenomicSet.read_parquet(input[0]).write_bed(output[0])


rule all_bed:
    input:
        expand(
            "results/bed/{g}/{region}.bed",
            region=config["functional_regions"],
            g=config["genome_subset_bed"],
        ),


# CDS, another version
rule intervals_recipe_v5:
    input:
        "results/intervals/cds/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v5/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256

        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# 5' UTR
rule intervals_recipe_v6:
    input:
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
    output:
        "results/intervals/recipe/v6/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256

        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# 3' UTR
rule intervals_recipe_v7:
    input:
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
    output:
        "results/intervals/recipe/v7/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256

        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# ncRNA exons
rule intervals_recipe_v8:
    input:
        "results/intervals/ncrna_exons/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
    output:
        "results/intervals/recipe/v8/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256

        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# promoters
rule intervals_recipe_v9:
    input:
        "results/intervals/promoters/256/256/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/ncrna_exons/{g}.parquet",
    output:
        "results/intervals/recipe/v9/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
            GenomicSet.read_parquet(input[5]),  # ncrna_exons
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals & defined
        intervals.write_bed(output[0])


# promoters (larger context)
rule intervals_recipe_v10:
    input:
        "results/intervals/promoters/2048/2048/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/ncrna_exons/{g}.parquet",
    output:
        "results/intervals/recipe/v10/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
            GenomicSet.read_parquet(input[5]),  # ncrna_exons
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals & defined
        intervals.write_bed(output[0])

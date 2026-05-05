"""Subset derivation for the zoonomia projection dataset (issue #149).

v1 = identity (no derivation rule needed; v1 source is the
all_species_with_sequence.parquet directly).
v2 = mRNA-TSS proximity (window overlaps [TSS - tss_flank, TSS + tss_flank]
     for any Ensembl protein_coding transcript).
"""

TSS_FLANK = int(config.get("tss_flank", 256))


rule derive_subset_v2_tss_mrna:
    """Generate v2 query_names: windows overlapping any mRNA TSS ± tss_flank."""
    input:
        gtf="results/annotation/Homo_sapiens.GRCh38.115.gtf.gz",
        bed="results/bed/min{min_p}.bed.gz",
    output:
        band="results/projection/min{min_p}/subsets_def/v2_band.bed",
        names="results/projection/min{min_p}/subsets_def/v2.query_names.txt",
    params:
        flank=TSS_FLANK,
    conda:
        "../envs/bioinformatics.yaml"
    run:
        from bolinas.projection.tss import write_mrna_tss_band_bed

        n_band = write_mrna_tss_band_bed(input.gtf, params.flank, output.band)
        # Ensembl rel 115 has ~90k protein_coding transcripts; merged bands
        # collapse closely-spaced TSSes — expect ~50k–80k merged intervals.
        assert n_band > 30_000, f"unexpectedly few mRNA TSS bands: {n_band}"
        shell(
            "bedtools intersect -u -a {input.bed} -b {output.band} "
            "| cut -f4 > {output.names}"
        )
        with open(output.names) as fh:
            kept = sum(1 for _ in fh)
        assert kept > 0, "v2 subset is empty"


# Override the existing PR-157 ``subset_dataset`` so derived subset
# definitions live under ``results/.../subsets_def/`` (Snakemake outputs)
# rather than ``config/subsets/`` (which is reserved for committed inputs).
ruleorder: subset_dataset_derived > subset_dataset


rule subset_dataset_derived:
    """Lazy-filter all_species_with_sequence to a derived subset."""
    input:
        all_species="results/projection/min{min_p}/all_species_with_sequence.parquet",
        names="results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
    output:
        "results/projection/min{min_p}/subsets/{subset}.parquet",
    threads: 1
    resources:
        mem_mb=4000,
    run:
        filter_to_subset(input.all_species, input.names, output[0])

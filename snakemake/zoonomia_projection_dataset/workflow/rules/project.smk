"""Cross-mammal projection step (issue #149, follows benchmark #153).

Projects the conservation-filtered human BED ``results/bed/min{p}.bed.gz``
onto every species in ``config/species_zoonomia_447_family_dedup.tsv`` via
``halLiftover --noDupes`` against the Zoonomia 447 Cactus HAL (1.18 TiB,
staged outside Snakemake to ``HAL_PATH`` by ``sky/project.yaml``). All
post-processing logic lives in ``src/bolinas/projection``.

DAG (per ``min_p``):

    results/bed/min{p}.bed.gz                (existing pipeline output)
        │
        │  prepare_input_ucsc — strip Ensembl bare names, prepend "chr",
        │                       emit BED6 with strand=+ and score=0
        ▼
    results/projection/input/min{p}.ucsc.bed
        │
        │  hal_chrom_sizes (per species)  +  project_one_species (║×SPECIES)
        ▼
    results/projection/min{p}/per_species/{species}.parquet
        │
        │  merge_projected
        ▼
    results/projection/min{p}/all_species.parquet

Smoke tier prepends two ZRS cCRE rows (issue #120) before head-truncating
to chr1, and gates on ``zrs_sanity_check``.
"""


from bolinas.projection.filter import filter_length, filter_single_chrom_strand
from bolinas.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
)
from bolinas.projection.resize import resize_to_length


# Two cCREs from SCREEN Registry V4 inside the canonical ZRS limb enhancer
# (hg38 chr7:156790115-156793672); Mus_musculus orthologs verified at
# mm10 chr5:29315086-29315432 / chr5:29316458-29316801. Used for the
# smoke-tier sanity check (zrs_sanity_check rule).
ZRS_BED6_LINES: list[str] = [
    "chr7\t156791361\t156791613\tzrs_EH38E2604086\t0\t+",
    "chr7\t156792617\t156792954\tzrs_EH38E2604087\t0\t+",
]

SOURCE_SPECIES = "Homo_sapiens"

PER_SPECIES_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.Utf8,
    "species": pl.Utf8,
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.Utf8,
    "t_src_size": pl.Int64,
}


rule prepare_input_ucsc:
    """Convert filtered BED to UCSC chrom names + BED6 (halLiftover input).

    The existing pipeline writes BEDs with bare Ensembl names ("1", "2", ...,
    "X", "Y") and 4 columns. halLiftover (Cactus) requires UCSC "chr1"/etc
    and preserves the input column count — BED6 with strand survives, BED4
    drops it (benchmark patch a84e063). For smoke tier, also prepend the
    ZRS cCRE rows and head-truncate the rest to chr1.
    """
    input:
        "results/bed/min{min_p}.bed.gz",
    output:
        "results/projection/input/min{min_p}.ucsc.bed",
    params:
        tier=TIER,
    run:
        df = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
            },
        )
        # Ensembl bare name → UCSC "chr"+name. Already-prefixed inputs would
        # double-prefix so guard with an assertion.
        assert (
            not df["chrom"].str.starts_with("chr").any()
        ), f"input BED has UCSC-style chroms; expected Ensembl bare: {df['chrom'].unique().to_list()[:5]}"

        with open(output[0], "w") as fout:
            if params.tier == "smoke":
                # ZRS cCREs first so they're guaranteed to be in the
                # head-truncated output regardless of the chr1 row count.
                for line in ZRS_BED6_LINES:
                    fout.write(line + "\n")
                df = df.filter(pl.col("chrom") == "1").head(1000)
            for row in df.iter_rows(named=True):
                # BED6: strand=+ (no strand info on tile windows), score=0.
                fout.write(
                    f"chr{row['chrom']}\t{row['start']}\t{row['end']}"
                    f"\t{row['name']}\t0\t+\n"
                )


rule hal_chrom_sizes:
    """Per-species `halStats --chromSizes` → 2-col TSV.

    Single-threaded; 48-way species parallelism. ~30 s/species cold-cache.
    """
    output:
        "results/projection/chrom_sizes/{species}.tsv",
    threads: 1
    resources:
        mem_mb=1500,
    shell:
        "halStats --chromSizes {wildcards.species} %s > {output}" % HAL_PATH


rule project_one_species:
    """Project the UCSC BED onto one species via halLiftover; filter; resize.

    Per (query, species) the final output has at most one row: raw
    ``halLiftover --noDupes`` may emit multiple rows when the query crosses
    alignment block boundaries on the same chrom+strand;
    ``filter_single_chrom_strand`` collapses those to a single merged span
    and drops multi-chrom/multi-strand groups. ``filter_length`` then drops
    merged spans outside ``[pre_resize_min_len, pre_resize_max_len]`` —
    typically 95-100 % of survivors are in the [128, 512] range per #153.
    """
    input:
        bed="results/projection/input/min{min_p}.ucsc.bed",
        chrom_sizes="results/projection/chrom_sizes/{species}.tsv",
    output:
        parquet="results/projection/min{min_p}/per_species/{species}.parquet",
    threads: 1
    resources:
        mem_mb=2000,
    run:
        species = wildcards.species
        # halLiftover writes a temp BED next to the input; keep it on the
        # same NVMe-mounted filesystem as the inputs.
        out_dir = Path(output.parquet).parent
        work_dir = out_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_bed = work_dir / f"{species}.bed"

        run_halliftover(
            HAL_PATH, SOURCE_SPECIES, input.bed, species, raw_bed, no_dupes=True
        )

        df = parse_halliftover_bed(raw_bed, species=species)
        df = attach_src_size(df, input.chrom_sizes)
        df = filter_single_chrom_strand(df)
        df = filter_length(df, min_len=PRE_RESIZE_MIN, max_len=PRE_RESIZE_MAX)
        df = df.filter(pl.col("t_src_size") >= TARGET_LEN)

        if df.is_empty():
            pl.DataFrame(schema=PER_SPECIES_SCHEMA).write_parquet(output.parquet)
            return

        new_starts: list[int] = []
        new_ends: list[int] = []
        for row in df.iter_rows(named=True):
            ns, ne = resize_to_length(
                row["t_start"], row["t_end"], TARGET_LEN, row["t_src_size"]
            )
            new_starts.append(ns)
            new_ends.append(ne)
        resized = pl.DataFrame(
            {
                "query_name": df["query_name"].to_list(),
                "species": df["species"].to_list(),
                "t_chrom": df["t_chrom"].to_list(),
                "t_start": new_starts,
                "t_end": new_ends,
                "t_strand": df["t_strand"].to_list(),
                "t_src_size": df["t_src_size"].to_list(),
            },
            schema=PER_SPECIES_SCHEMA,
        )

        # Loud invariants — reproducibility > silent corruption (CLAUDE.md).
        assert (resized["t_end"] - resized["t_start"] == TARGET_LEN).all()
        assert (resized["t_start"] >= 0).all()
        assert (resized["t_end"] <= resized["t_src_size"]).all()
        assert (
            resized["query_name"].n_unique() == resized.height
        ), "expected at most one row per query_name after filter"

        resized.write_parquet(output.parquet)
        raw_bed.unlink(missing_ok=True)


rule merge_projected:
    """Concatenate per-species Parquets into one all-species Parquet (streaming)."""
    input:
        lambda wc: expand(
            "results/projection/min{min_p}/per_species/{species}.parquet",
            min_p=[wc.min_p],
            species=SPECIES,
        ),
    output:
        "results/projection/min{min_p}/all_species.parquet",
    resources:
        mem_mb=4000,
    run:
        lf = pl.concat([pl.scan_parquet(p) for p in input], how="vertical")
        lf.sink_parquet(output[0])


rule zrs_sanity_check:
    """Smoke-tier-only sanity gate: ZRS cCREs lift human→mouse correctly.

    Not part of the full-tier DAG; ``rule all_projected`` only depends on
    this when ``TIER == "smoke"``. See ``scripts/zrs_sanity_check.py``.
    """
    input:
        mus="results/projection/min{min_p}/per_species/Mus_musculus.parquet",
        script=workflow.source_path("../../scripts/zrs_sanity_check.py"),
    output:
        "results/projection/min{min_p}/zrs_sanity.txt",
    shell:
        "python {input.script} --mus-parquet {input.mus} --output {output}"


rule all_projected:
    """Final target for the cross-mammal projection step.

    Smoke tier additionally requires the ZRS sanity check to pass; that
    rule's own ``shell`` block exits 1 on any FAIL and so fails this rule
    transitively. Full tier skips the sanity check (no ZRS rows in the input
    BED — they're only added in smoke).
    """
    input:
        ([
            f"results/projection/min{PROJECT_MIN_P}/all_species.parquet",
            f"results/projection/min{PROJECT_MIN_P}/zrs_sanity.txt",
        ] if TIER == "smoke" else [
            f"results/projection/min{PROJECT_MIN_P}/all_species.parquet",
        ]),

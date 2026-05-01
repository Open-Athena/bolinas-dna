rule ldscore_download:
    output:
        local(temp(directory("results/ldscore/UKBB.EUR.ldscore.ht"))),
    params:
        s3=config["complex_traits"]["ldscore_s3"],
    shell:
        "aws s3 cp --no-sign-request --recursive {params.s3} {output}"


rule ldscore_convert:
    input:
        "results/ldscore/UKBB.EUR.ldscore.ht",
    output:
        local(temp("results/ldscore/UKBB.EUR.ldscore.tsv.bgz")),
    shell:
        r"""
        INPUT_ABS=$(readlink -f {input})
        OUTPUT_DIR_ABS=$(readlink -f $(dirname {output}))
        OUTPUT_NAME=$(basename {output})
        HOST_UID=$(id -u)
        HOST_GID=$(id -g)
        # Run Hail as root inside the container (Spark/Ivy needs a real $HOME
        # with a /etc/passwd entry; --user $(id -u):$(id -g) breaks that and
        # crashes with "basedir must be absolute: ?/.ivy2/local"). Chown the
        # output back to the host user so snakemake can manage it afterwards.
        docker run --rm \
            -v "$INPUT_ABS":/data/input:ro \
            -v "$OUTPUT_DIR_ABS":/data/out \
            hailgenetics/hail:0.2.130.post1-py3.11 \
            bash -c "python3 -c \"import hail as hl; ht = hl.read_table('/data/input'); print(ht.describe()); ht.export('/data/out/$OUTPUT_NAME')\" && chown -R $HOST_UID:$HOST_GID /data/out"
        """


rule ldscore_process:
    input:
        "results/ldscore/UKBB.EUR.ldscore.tsv.bgz",
    output:
        "results/ldscore/UKBB.EUR.ldscore.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                separator="\t",
                columns=["locus", "alleles", "AF", "ld_score"],
            )
            .with_columns(
                pl.col("locus")
                .str.split_exact(":", 1)
                .struct.rename_fields(["chrom", "pos"]),
                pl.col("alleles")
                .str.replace("[", "", literal=True)
                .str.replace("]", "", literal=True)
                .str.replace_all('"', "", literal=True)
                .str.split_exact(",", 1)
                .struct.rename_fields(["ref", "alt"]),
                pl.when(pl.col("AF") < 0.5)
                .then(pl.col("AF"))
                .otherwise(1 - pl.col("AF"))
                .alias("MAF"),
            )
            .with_columns(
                pl.col("locus").struct.field("chrom"),
                pl.col("locus").struct.field("pos").cast(int),
                pl.col("alleles").struct.field("ref"),
                pl.col("alleles").struct.field("alt"),
            )
            .drop(["locus", "alleles"])
            .pipe(filter_snp)
            .select(COORDINATES + ["MAF", "ld_score"])
            .sort(COORDINATES)
            .write_parquet(output[0])
        )

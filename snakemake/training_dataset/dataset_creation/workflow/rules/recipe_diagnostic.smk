# Diagnostic comparing interval recipes v20 (segmentation) vs v30 (projection)
# in human and mouse, to investigate the 2.3× distal-AUPRC gap reported in
# issue #136. See src/bolinas/diagnostics/recipe_compare.py for the metrics.

from bolinas.diagnostics.recipe_compare import (
    compute_recipe_summary,
    compute_seg_quantile_sweep,
)

HUMAN_GENOME = "GCF_000001405.40"
MOUSE_GENOME = "GCF_000001635.27"


# phastCons-43p (Zoonomia 43-primate phastCons; threshold 0.961 calibrated to
# ~3.5% conserved-base fraction matching phyloP-241way 2.27).
# NOTE: PR #141 also adds this rule in stats.smk — dedup on merge.
rule download_phastcons_43p_for_diagnostic:
    output:
        "results/conservation/phastCons_43p.bw",
    shell:
        "wget -O {output} https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig"


rule recipe_diagnostic_human:
    input:
        v20=f"results/intervals/recipe/v20/{HUMAN_GENOME}.bed.gz",
        v30=f"results/intervals/recipe/v30/{HUMAN_GENOME}.bed.gz",
        ELS="results/cre/ELS.parquet",
        ELS_conserved="results/cre/ELS_conserved_20.parquet",
        promoters=f"results/intervals/promoters/2048/2048/{HUMAN_GENOME}.parquet",
        twobit=f"results/genome/{HUMAN_GENOME}.2bit",
        scannable=f"results/intervals/scannable/{HUMAN_GENOME}.bed.gz",
        exons=f"results/intervals/exons/{HUMAN_GENOME}.parquet",
        phylop="results/conservation/cactus241way.phyloP.bw",
        phastcons="results/conservation/phastCons_43p.bw",
        chrom_map=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/diagnostics/recipe_v20_v30/{HUMAN_GENOME}.parquet",
    run:
        chrom_map_df = pl.read_csv(input.chrom_map, separator="\t")
        refseq_to_ucsc = dict(zip(chrom_map_df["refseq"], chrom_map_df["ucsc"]))
        # cCRE parquet uses bare chrom names ('1', 'X'); recipe BEDs use RefSeq.
        # Build bare → RefSeq from the canonical mapping by stripping 'chr'.
        bare_to_refseq = {
            ucsc.removeprefix("chr"): refseq for refseq, ucsc in refseq_to_ucsc.items()
        }
        df = compute_recipe_summary(
            v20_bed=input.v20,
            v30_bed=input.v30,
            twobit=input.twobit,
            promoters_parquet=input.promoters,
            ccre_paths={"ELS": input.ELS, "ELS_conserved_20": input.ELS_conserved},
            ccre_chrom_map=bare_to_refseq,
            scannable_bed=input.scannable,
            exons_parquet=input.exons,
            conservation_tracks={
                "phylop_241m": (input.phylop, 2.27),
                "phastcons_43p": (input.phastcons, 0.961),
            },
            chrom_map=refseq_to_ucsc,
        )
        df["species"] = "homo_sapiens"
        df["genome"] = HUMAN_GENOME
        # Range asserts on bigwig-derived metrics (loud failure near the bug).
        for metric in [
            "frac_phylop_241m_ge_threshold",
            "frac_phastcons_43p_ge_threshold",
        ]:
            sub = df.loc[df["metric"] == metric, "value"]
            assert sub.between(0, 1).all(), f"{metric} out of [0,1]: {sub.tolist()}"
        mean_phylop = df.loc[df["metric"] == "mean_phylop_241m", "value"]
        assert mean_phylop.between(
            -20, 10
        ).all(), f"mean_phylop_241m out of range: {mean_phylop.tolist()}"
        mean_phastcons = df.loc[df["metric"] == "mean_phastcons_43p", "value"]
        assert mean_phastcons.between(
            0, 1
        ).all(), f"mean_phastcons_43p out of [0,1]: {mean_phastcons.tolist()}"
        df.to_parquet(output[0], index=False)


rule recipe_diagnostic_mouse:
    input:
        v20=f"results/intervals/recipe/v20/{MOUSE_GENOME}.bed.gz",
        v30=f"results/intervals/recipe/v30/{MOUSE_GENOME}.bed.gz",
        promoters=f"results/intervals/promoters/2048/2048/{MOUSE_GENOME}.parquet",
        twobit=f"results/genome/{MOUSE_GENOME}.2bit",
        exons=f"results/intervals/exons/{MOUSE_GENOME}.parquet",
    output:
        f"results/diagnostics/recipe_v20_v30/{MOUSE_GENOME}.parquet",
    run:
        df = compute_recipe_summary(
            v20_bed=input.v20,
            v30_bed=input.v30,
            twobit=input.twobit,
            promoters_parquet=input.promoters,
            exons_parquet=input.exons,
            ccre_paths=None,  # mouse cCRE is mm10 + needs liftover; skip in v1
            conservation_tracks=None,  # phyloP/phastCons are hg38-only or need liftover
        )
        df["species"] = "mus_musculus"
        df["genome"] = MOUSE_GENOME
        df.to_parquet(output[0], index=False)


rule seg_quantile_sweep_human:
    """Sweep the segmentation top-quantile threshold (1%, 2%, 3%, 4%, 5%)
    and characterize each resulting interval set the same way as
    recipe_diagnostic_human, plus a v30 reference row. Issue #143 follow-up."""
    input:
        predictions=f"results/enhancer_predictions_segmentation/{HUMAN_GENOME}.parquet",
        exons=f"results/intervals/exons/{HUMAN_GENOME}.parquet",
        defined=f"results/intervals/defined/{HUMAN_GENOME}.bed.gz",
        scannable=f"results/intervals/scannable/{HUMAN_GENOME}.bed.gz",
        v30=f"results/intervals/recipe/v30/{HUMAN_GENOME}.bed.gz",
        ELS="results/cre/ELS.parquet",
        ELS_conserved="results/cre/ELS_conserved_20.parquet",
        promoters=f"results/intervals/promoters/2048/2048/{HUMAN_GENOME}.parquet",
        twobit=f"results/genome/{HUMAN_GENOME}.2bit",
        phylop="results/conservation/cactus241way.phyloP.bw",
        phastcons="results/conservation/phastCons_43p.bw",
        chrom_map=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/diagnostics/seg_quantile_sweep/{HUMAN_GENOME}.parquet",
    run:
        chrom_map_df = pl.read_csv(input.chrom_map, separator="\t")
        refseq_to_ucsc = dict(zip(chrom_map_df["refseq"], chrom_map_df["ucsc"]))
        bare_to_refseq = {
            ucsc.removeprefix("chr"): refseq for refseq, ucsc in refseq_to_ucsc.items()
        }
        df = compute_seg_quantile_sweep(
            predictions_parquet=input.predictions,
            exons_parquet=input.exons,
            defined_bed=input.defined,
            quantiles=[0.01, 0.02, 0.03, 0.04, 0.05],
            target_size=255,
            ccre_paths={"ELS": input.ELS, "ELS_conserved_20": input.ELS_conserved},
            ccre_chrom_map=bare_to_refseq,
            scannable_bed=input.scannable,
            promoters_parquet=input.promoters,
            twobit=input.twobit,
            conservation_tracks={
                "phylop_241m": (input.phylop, 2.27),
                "phastcons_43p": (input.phastcons, 0.961),
            },
            chrom_map=refseq_to_ucsc,
            v30_bed=input.v30,
        )
        df["species"] = "homo_sapiens"
        df["genome"] = HUMAN_GENOME
        df = df[["species", "genome", "recipe", "metric", "value"]]
        df.to_parquet(output[0], index=False)


rule recipe_diagnostic_summary:
    input:
        f"results/diagnostics/recipe_v20_v30/{HUMAN_GENOME}.parquet",
        f"results/diagnostics/recipe_v20_v30/{MOUSE_GENOME}.parquet",
    output:
        "results/diagnostics/recipe_v20_v30/summary.parquet",
    run:
        dfs = [pd.read_parquet(p) for p in input]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined[["species", "genome", "recipe", "metric", "value"]]
        combined.to_parquet(output[0], index=False)

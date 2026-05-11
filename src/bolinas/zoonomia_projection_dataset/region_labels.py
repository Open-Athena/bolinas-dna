"""Per-anchor region-type annotation for the zoonomia projection dataset.

Labels every conservation-filtered human anchor (255 bp) with exactly one of:

    cds  >  utr3  >  ncrna_exon  >  tss_region_and_utr5  >  ccre_non_promoter  >  background

(see ``REGION_LABELS`` for the functional five; ``BACKGROUND`` covers the
rest). The label is the highest-priority region with any overlap, *provided*
the union-of-functional fraction over the window is ≥ ``functional_threshold``;
otherwise the window is ``background``.

``ccre_non_promoter`` = every ENCODE cCRE V4 class **except** ``PLS``
(so: dELS, pELS, CA, CA-CTCF, CA-TF, CA-H3K4me3, TF), extended by
``ccre_flank`` bp on each side. PLS-overlapping windows near an annotated
TSS are captured by ``tss_region_and_utr5`` instead.

All extractors are Ensembl-flavored: ``transcript_biotype "protein_coding"``,
``"lncRNA"``, etc. RefSeq-flavored helpers in ``bolinas.data.utils``
(``get_mrna_exons``, ``get_5_prime_utr``, ``get_3_prime_utr``,
``get_ncrna_exons``, ``get_promoters``) must not be used here — they filter
on RefSeq biotype vocabulary (``"mRNA"``, ``"lnc_RNA"``, ...). The
``_assert_ensembl_gtf`` check at the top of ``build_region_beds`` guards
against an accidental RefSeq GTF.
"""

from pathlib import Path

import bioframe as bf
import numpy as np
import pandas as pd
import polars as pl

from bolinas.data.intervals import GenomicSet
from bolinas.data.utils import (
    get_cds,
    get_exons,
    get_promoters_from_exons,
    load_annotation,
)
from bolinas.projection.tss import get_ensembl_protein_coding_exons
from bolinas.zoonomia_projection_dataset.validation import (
    get_ensembl_3_prime_utr,
    get_ensembl_5_prime_utr,
)

# Functional region labels in canonical (default-priority) order.
REGION_LABELS: tuple[str, ...] = (
    "cds",
    "utr3",
    "ncrna_exon",
    "tss_region_and_utr5",
    "ccre_non_promoter",
)
BACKGROUND_LABEL = "background"


# ============================================================================
# Ensembl-only extractors (no RefSeq fallback)
# ============================================================================


def get_ensembl_all_transcript_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """Every exon row carrying a transcript_id, no biotype filter.

    Mirrors :func:`bolinas.projection.tss.get_ensembl_protein_coding_exons`
    but drops the ``transcript_biotype == "protein_coding"`` filter — keeps
    every annotated transcript (mRNA, lncRNA, miRNA, snoRNA, pseudogenes,
    retained_intron, NMD, ...). Used to derive the TSS band for
    ``tss_region_and_utr5``.

    Returns ``[chrom, start, end, strand, transcript_id]`` — the shape
    :func:`bolinas.data.utils.get_promoters_from_exons` expects.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
        )
        .filter(pl.col("transcript_id").is_not_null())
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )


def get_ensembl_gene_body(ann: pl.DataFrame) -> GenomicSet:
    """Union of ``feature == "gene"`` spans (diagnostic only).

    Used to compute ``gene_body_frac`` / ``intron_frac`` /
    ``intergenic_frac`` columns. Does not feed the label.
    """
    return GenomicSet(ann.filter(pl.col("feature") == "gene"))


def _assert_ensembl_gtf(ann: pl.DataFrame) -> None:
    """Crash loudly if the GTF isn't Ensembl-flavored.

    RefSeq GTFs use ``transcript_biotype "mRNA"``; Ensembl uses
    ``"protein_coding"``. The extractors in this module *require* Ensembl
    vocabulary — running them on a RefSeq GTF silently returns empty
    ``cds`` / ``utr3`` / ``utr5`` sets without this check.
    """
    n_pc = (
        ann.filter(pl.col("feature") == "transcript")
        .filter(
            pl.col("attribute").str.contains(
                'transcript_biotype "protein_coding"', literal=True
            )
        )
        .height
    )
    assert n_pc > 0, (
        'no transcripts with transcript_biotype "protein_coding" — '
        "this pipeline requires an Ensembl-flavored GTF. RefSeq uses "
        'transcript_biotype "mRNA"; do not mix vocabularies.'
    )


# ============================================================================
# Region BED construction
# ============================================================================


def build_region_beds(
    ann_path: str | Path,
    cre_parquet: str | Path,
    defined: GenomicSet,
    *,
    tss_radius: int,
    ccre_flank: int,
) -> dict[str, GenomicSet]:
    """Build the 5 functional region BEDs + 2 diagnostic sets.

    Every output GenomicSet is intersected with ``defined`` (genome minus
    N regions) at the end, so per-window fractions never count bases that
    fall in unsequenceable masked regions.

    Args:
        ann_path: Ensembl GTF (release pinned in the calling pipeline).
        cre_parquet: ENCODE cCRE V4 parquet with ``chrom, start, end,
            cre_class`` columns, produced by ``validation.smk:cre_process``.
        defined: ``genome − N`` GenomicSet from ``windows.smk:undefined``.
        tss_radius: ± bp around every transcript's TSS for the
            ``tss_region_and_utr5`` class.
        ccre_flank: bp added on each side of every non-PLS cCRE before it
            contributes to the ``ccre_non_promoter`` class.

    Returns:
        Dict with keys ``cds``, ``utr3``, ``ncrna_exon``,
        ``tss_region_and_utr5``, ``ccre_non_promoter`` (the five
        functional labels) plus ``gene_body`` and ``all_exons``
        (diagnostic, for intron / intergenic decomposition).
    """
    ann = load_annotation(str(ann_path))
    _assert_ensembl_gtf(ann)

    cds = get_cds(ann)
    utr3 = get_ensembl_3_prime_utr(ann)
    utr5 = get_ensembl_5_prime_utr(ann)

    pc_exons = GenomicSet(
        get_ensembl_protein_coding_exons(ann).select(["chrom", "start", "end"])
    )
    all_exons = get_exons(ann)
    ncrna_exon = all_exons - pc_exons

    all_tx_exons = get_ensembl_all_transcript_exons(ann)
    assert len(all_tx_exons) > 0, "GTF has no transcript-tagged exon rows"
    tss_band = get_promoters_from_exons(
        all_tx_exons, n_upstream=tss_radius, n_downstream=tss_radius
    )
    tss_region_and_utr5 = tss_band | utr5

    ccre_df = pl.read_parquet(cre_parquet).filter(pl.col("cre_class") != "PLS")
    assert len(ccre_df) > 0, (
        f"no non-PLS cCREs in {cre_parquet} — wrong file or unexpected schema?"
    )
    ccre_non_promoter = (
        GenomicSet(ccre_df.select(["chrom", "start", "end"])).add_flank(ccre_flank)
    )

    gene_body = get_ensembl_gene_body(ann)

    return {
        "cds": cds & defined,
        "utr3": utr3 & defined,
        "ncrna_exon": ncrna_exon & defined,
        "tss_region_and_utr5": tss_region_and_utr5 & defined,
        "ccre_non_promoter": ccre_non_promoter & defined,
        "gene_body": gene_body & defined,
        "all_exons": all_exons & defined,
    }


# ============================================================================
# Per-window labeling
# ============================================================================


def _coverage_bp(windows: pd.DataFrame, region: pd.DataFrame) -> np.ndarray:
    """Per-window basepair coverage by ``region``, computed chrom-by-chrom.

    Returns an array aligned to ``windows.index`` (caller must reset index
    before passing in). Empty ``region`` yields all zeros.

    Caveat: ``bf.coverage`` resets its input DataFrame's index in place to
    ``[0, 1, ...]`` (it sorts internally for the overlap join). We pass a
    ``.copy()`` and capture ``sub.index`` *before* the call so each chrom
    iteration writes coverage to the correct rows of ``out``. Without this,
    the second-and-later chroms in the groupby loop silently corrupt
    earlier chroms' values — the bug surfaces only on multi-chrom inputs.
    """
    assert windows.index.is_monotonic_increasing and windows.index[0] == 0, (
        "_coverage_bp requires reset_index() windows DataFrame"
    )
    out = np.zeros(len(windows), dtype=np.int64)
    if len(region) == 0:
        return out
    region_by_chrom = {chrom: sub for chrom, sub in region.groupby("chrom", sort=False)}
    for chrom, sub in windows.groupby("chrom", sort=False):
        chrom_region = region_by_chrom.get(chrom)
        if chrom_region is None or len(chrom_region) == 0:
            continue
        orig_idx = sub.index.to_numpy()  # Capture BEFORE bf.coverage mutates sub.index.
        cov = (
            bf.coverage(sub.copy(), chrom_region, return_input=False)["coverage"]
            .to_numpy()
        )
        out[orig_idx] = cov
    return out


def label_windows(
    windows_bed: str | Path,
    beds: dict[str, GenomicSet],
    *,
    functional_threshold: float,
    priority: list[str],
) -> pl.DataFrame:
    """Label every window with one of ``REGION_LABELS`` or ``"background"``.

    Args:
        windows_bed: BED4 (chrom, start, end, name) of human anchors, e.g.
            ``results/human/intervals/filtered/min0.20.bed.gz``. .gz auto-
            detected by pandas.
        beds: dict from :func:`build_region_beds` containing the 5
            functional regions + ``gene_body`` + ``all_exons``.
        functional_threshold: window's union-of-functional fraction must
            be ≥ this to escape ``background``. Range [0, 1].
        priority: ordering of functional labels for tie-breaking when a
            window overlaps multiple regions. Must be a permutation of
            ``REGION_LABELS``.

    Returns:
        Polars DataFrame with one row per input window and columns
        ``name, chrom, start, end, label, functional_frac, cds_frac,
        utr3_frac, ncrna_exon_frac, tss_region_and_utr5_frac,
        ccre_non_promoter_frac, gene_body_frac, intron_frac,
        intergenic_frac``.
    """
    assert 0.0 <= functional_threshold <= 1.0, functional_threshold
    assert set(priority) == set(REGION_LABELS), (
        f"priority {priority!r} must be a permutation of {list(REGION_LABELS)!r}"
    )
    missing_functional = set(REGION_LABELS) - set(beds.keys())
    assert not missing_functional, f"beds missing functional keys: {missing_functional}"
    for diag in ("gene_body", "all_exons"):
        assert diag in beds, f"beds missing diagnostic key: {diag!r}"

    windows = pd.read_csv(
        str(windows_bed),
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name"],
        dtype={"chrom": str},
    ).reset_index(drop=True)
    sizes = (windows["end"] - windows["start"]).to_numpy()
    assert (sizes > 0).all(), "non-positive window sizes"

    coords = windows[["chrom", "start", "end"]]

    # Per-region overlap fractions for the 5 functional labels.
    frac: dict[str, np.ndarray] = {}
    for label in REGION_LABELS:
        cov_bp = _coverage_bp(coords, beds[label].to_pandas())
        frac[label] = cov_bp / sizes

    # Union of all 5 functional regions — basis for the threshold.
    functional_union = beds[REGION_LABELS[0]]
    for label in REGION_LABELS[1:]:
        functional_union = functional_union | beds[label]
    functional_frac = _coverage_bp(coords, functional_union.to_pandas()) / sizes

    # Diagnostic: gene body, intron, intergenic.
    gene_body_frac = _coverage_bp(coords, beds["gene_body"].to_pandas()) / sizes
    exon_frac = _coverage_bp(coords, beds["all_exons"].to_pandas()) / sizes
    intron_frac = np.clip(gene_body_frac - exon_frac, a_min=0.0, a_max=None)
    intergenic_frac = np.clip(1.0 - gene_body_frac, a_min=0.0, a_max=None)

    # Priority-walk: pick the first label in ``priority`` that has any overlap.
    n = len(windows)
    label_arr = np.full(n, BACKGROUND_LABEL, dtype=object)
    is_functional = functional_frac >= functional_threshold
    unassigned = is_functional.copy()
    for label in priority:
        pick = unassigned & (frac[label] > 0)
        label_arr[pick] = label
        unassigned &= ~pick
        if not unassigned.any():
            break
    # Anything still unassigned among functional windows (functional_frac ≥
    # threshold but no individual fraction > 0) cannot occur — the union is
    # the union of these very same sets. Assert as an invariant.
    assert not unassigned.any(), (
        "labeler bug: window passed functional_threshold but no per-region frac > 0"
    )

    return pl.DataFrame(
        {
            "name": windows["name"].to_numpy(),
            "chrom": windows["chrom"].to_numpy(),
            "start": windows["start"].to_numpy(),
            "end": windows["end"].to_numpy(),
            "label": label_arr.astype(str),
            "functional_frac": functional_frac,
            **{f"{label}_frac": frac[label] for label in REGION_LABELS},
            "gene_body_frac": gene_body_frac,
            "intron_frac": intron_frac,
            "intergenic_frac": intergenic_frac,
        }
    )

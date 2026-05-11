"""Generate leaderboard markdown tables for issues #161 / #162 (+ new eqtl).

Pulls per-(method, dataset, subset) PairwiseAccuracy + SE from S3:
  - conservation_eval: 7 conservation tracks
  - evals_v2: 5 model checkpoints
  - alphagenome_eval: AlphaGenome variant scorer

Combines into one table per dataset. n_pairs ≥ 30 cutoff. Bold = top method
per subset + any method within 0.01 of the top.

Outputs three markdown chunks ready to drop into the leaderboard issues.
"""
from __future__ import annotations

import polars as pl

# Per-dataset score_type per pipeline.
SCORE_TYPE = {
    "evals_v2": {
        "mendelian_traits": "minus_llr",
        "complex_traits": "abs_llr",
        "eqtl": "abs_llr",
    },
    "conservation": "score",
    "alphagenome": "alphagenome_max_l2",
}

EVALS_V2_MODELS = [
    ("exp55-mammals", "promoters, mammals"),
    ("exp58-mammals", "CDS, mammals"),
    ("exp58-animals", "CDS, animals"),
    ("exp59-mammals", "downstream, mammals"),
    ("exp136-proj_v30", "enhancers, mammals"),
]
CONSERVATION_TRACKS = [
    "phastCons_100v",
    "phastCons_43p",
    "phastCons_470m",
    "phyloP_100v",
    "phyloP_241m",
    "phyloP_447m",
    "phyloP_470m",
]

SUBSET_DISPLAY = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5' UTR",
    "distal": "Distal",
    "3_prime_UTR_variant": "3' UTR",
    "tss_proximal": "Promoter",
    "non_coding_transcript_exon_variant": "ncRNA",
    "synonymous_variant": "Synonymous",
}

DATASETS = ("mendelian_traits", "complex_traits", "eqtl")
S3 = "s3://oa-bolinas"
SPLIT = "train"
N_MIN = 30


def fmt(value: float, se: float) -> str:
    return f"{value:.3f} ± {se:.3f}"


def gather_methods(dataset: str) -> list[tuple[str, str | None, pl.DataFrame]]:
    """Return [(method_name, comment, per_subset_df), ...] for one dataset."""
    rows: list[tuple[str, str | None, pl.DataFrame]] = []

    # 1. conservation tracks
    cons = pl.read_parquet(
        f"{S3}/snakemake/conservation_eval/results/{dataset}/metrics_{SPLIT}.parquet"
    )
    for track in CONSERVATION_TRACKS:
        df = cons.filter(pl.col("score_name") == track).select(
            ["subset", "value", "se", "n_pairs"]
        )
        rows.append((f"`{track}`", None, df))

    # 2. evals_v2 models
    sct = SCORE_TYPE["evals_v2"][dataset]
    for model, comment in EVALS_V2_MODELS:
        df = pl.read_parquet(
            f"{S3}/snakemake/analysis/evals_v2/results/metrics/{model}/{dataset}.parquet"
        )
        df = df.filter(pl.col("score_type") == sct).filter(pl.col("split") == SPLIT)
        df = df.select(["subset", "value", "se", "n_pairs"])
        rows.append((f"`{model}`", comment, df))

    # 3. alphagenome
    try:
        ag = pl.read_parquet(
            f"{S3}/snakemake/alphagenome_eval/results/metrics/{dataset}.parquet"
        )
        ag = ag.filter(pl.col("score_type") == SCORE_TYPE["alphagenome"]).filter(
            pl.col("split") == SPLIT
        )
        rows.append(
            (
                "`AlphaGenome`",
                "variant scorer, API",
                ag.select(["subset", "value", "se", "n_pairs"]),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! alphagenome metrics missing for {dataset}: {exc}")

    return rows


def build_table(dataset: str) -> str:
    rows = gather_methods(dataset)

    # Subsets present in *any* method, with n_pairs ≥ 30, ordered by total n.
    subset_n: dict[str, int] = {}
    for _, _, df in rows:
        for s, n in df.select(["subset", "n_pairs"]).iter_rows():
            subset_n[s] = max(subset_n.get(s, 0), int(n))
    subsets = [
        s
        for s, n in sorted(subset_n.items(), key=lambda kv: -kv[1])
        if n >= N_MIN and s in SUBSET_DISPLAY
    ]
    if not subsets:
        return f"# {dataset}\n\nNo subset has n_pairs ≥ {N_MIN}.\n"

    # Per-subset top-value within rows (for bolding).
    top: dict[str, float] = {}
    cell_pa: dict[tuple[str, str], tuple[float, float]] = {}
    for method, _, df in rows:
        for s, v, se, _ in df.iter_rows():
            if s in subsets:
                cell_pa[(method, s)] = (v, se)
                top[s] = max(top.get(s, -1), v)

    header = (
        "| method | "
        + " | ".join(f"{SUBSET_DISPLAY[s]}<br>(n={subset_n[s]})" for s in subsets)
        + " |"
    )
    sep = "|---|" + "|".join("---" for _ in subsets) + "|"
    lines = [header, sep]
    for method, comment, df in rows:
        label = method + (f" ({comment})" if comment else "")
        cells = []
        for s in subsets:
            if (method, s) not in cell_pa:
                cells.append("—")
                continue
            v, se = cell_pa[(method, s)]
            text = fmt(v, se)
            if v >= top[s] - 0.01:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


for ds in DATASETS:
    print(f"\n{'#' * 70}\n# {ds}\n{'#' * 70}\n")
    print(build_table(ds))

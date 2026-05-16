"""Aggregate matched-pair evaluation metrics from S3 into leaderboard rows.

Reads pre-computed PairwiseAccuracy parquets emitted by these snakemake
pipelines, then collapses them into one ``MethodMetrics`` object per registered
method × dataset:

  - ``snakemake/analysis/evals_v2/``  → one parquet per ``(model, dataset)``,
    filter by ``score_type`` + ``split``.
  - ``snakemake/conservation_eval/``  → one parquet per ``(dataset, split)``,
    filter by ``score_name`` (the track).
  - ``snakemake/alphagenome_eval/``   → one parquet per dataset, filter by
    ``score_type`` + ``split``.
  - ``snakemake/gpn_star_eval/``      → one parquet per dataset, filter by
    ``score_type`` + ``split`` + ``model``.

Method registry (display name, family, training metadata, etc.) lives in
``dashboard/methods.yaml`` and is loaded via ``methods.load_methods``. This
module is the data layer for both the dashboard (``dashboard/``) and the
legacy issue-body patching CLI (``snakemake/evals/scratch/leaderboard_gen.py``).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Literal

import polars as pl

from bolinas.pipelines.evals.gpn_star import GPN_STAR_SCORE_COLUMN
from bolinas.pipelines.evals.methods import Method, methods_for_dataset
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET

S3 = "s3://oa-bolinas"
SPLIT = "train"
N_MIN = 30

# Per-family score-column lookup. Drives the ``score_type`` filter on the
# pipeline's metrics parquet. Most families use a single column across all
# three datasets; bolinas + gpn_star vary by dataset (different score
# definitions per benchmark type).
_FAMILY_SCORE_TYPE: dict[str, str | dict[str, str]] = {
    "bolinas": {
        "mendelian_traits": "minus_llr",
        "complex_traits": "abs_llr",
        "eqtl": "abs_llr",
    },
    "conservation": "score",
    "alphagenome": "alphagenome_max_l2",
    "gpn_star": GPN_STAR_SCORE_COLUMN,
}

SortAxis = Literal["macro", "global"]

# Which aggregate is the headline for each dataset — controls (a) the sort
# axis (top of table = top method on this aggregate) and (b) which aggregate
# column appears leftmost. Mendelian uses Macro Avg because the variant
# composition is ~92% missense (a ClinVar annotator-history artifact, not
# pathogenicity reality), so Global PA over-weights methods specialized for
# protein-coding variant interpretation. Complex/eqtl have very different
# subset compositions where the same bias doesn't apply, so they stay on Global.
LEADING_AGGREGATE: dict[str, SortAxis] = {
    "mendelian_traits": "macro",
    "complex_traits": "global",
    "eqtl": "global",
}

SUBSET_DISPLAY: dict[str, str] = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5' UTR",
    "distal": "Distal",
    "3_prime_UTR_variant": "3' UTR",
    "tss_proximal": "Promoter",
    "non_coding_transcript_exon_variant": "ncRNA",
    "synonymous_variant": "Synonymous",
}

DATASET_ISSUE: dict[str, int] = {
    "mendelian_traits": 161,
    "complex_traits": 162,
    "eqtl": 172,
}


def _score_type(family: str, dataset: str) -> str:
    st = _FAMILY_SCORE_TYPE[family]
    return st if isinstance(st, str) else st[dataset]


def _storage_options() -> dict[str, str] | None:
    """Toggle anonymous S3 reads via ``BOLINAS_S3_ANON=1``.

    Lets the GitHub Action build against a public-read bucket prefix
    without requiring AWS credentials. With anything else (default), polars
    walks the standard credential chain (env vars → ~/.aws → IMDS)."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return {"aws_skip_signature": "true", "aws_region": "us-east-2"}
    return None


@functools.lru_cache(maxsize=None)
def _read_parquet(path: str) -> pl.DataFrame:
    """Cached S3 read so families that share a per-dataset parquet
    (conservation, alphagenome, gpn_star) only fetch once per process."""
    return pl.read_parquet(path, storage_options=_storage_options())


def _parquet_path(method: Method, dataset: str) -> str:
    match method.family:
        case "bolinas":
            return (
                f"{S3}/snakemake/analysis/evals_v2/results/metrics/"
                f"{method.id}/{dataset}.parquet"
            )
        case "conservation":
            return (
                f"{S3}/snakemake/conservation_eval/results/"
                f"{dataset}/metrics_{SPLIT}.parquet"
            )
        case "alphagenome":
            return f"{S3}/snakemake/alphagenome_eval/results/metrics/{dataset}.parquet"
        case "gpn_star":
            return f"{S3}/snakemake/gpn_star_eval/results/metrics/{dataset}.parquet"
        case _:
            raise ValueError(f"unknown family {method.family!r}")


def fetch_method_metrics(method: Method, dataset: str) -> pl.DataFrame:
    """Return per-method, per-dataset rows with columns
    ``[subset, value, se, n_pairs, n_ties]`` — including the
    ``_global_`` and ``_macro_avg_`` aggregate rows."""
    assert dataset in method.datasets, (
        f"{method.id!r} is not registered for dataset {dataset!r}"
    )
    path = _parquet_path(method, dataset)
    df = _read_parquet(path)
    sct = _score_type(method.family, dataset)
    match method.family:
        case "bolinas":
            df = df.filter(pl.col("score_type") == sct).filter(pl.col("split") == SPLIT)
        case "conservation":
            df = df.filter(pl.col("score_name") == method.id)
        case "alphagenome":
            df = df.filter(pl.col("score_type") == sct).filter(pl.col("split") == SPLIT)
        case "gpn_star":
            df = (
                df.filter(pl.col("score_type") == sct)
                .filter(pl.col("split") == SPLIT)
                .filter(pl.col("model") == method.id)
            )
            assert df.height > 0, (
                f"no rows for GPN-Star method {method.id!r} on {dataset!r} in {path}"
            )
        case _:
            raise ValueError(f"unknown family {method.family!r}")
    return df.select(["subset", "value", "se", "n_pairs", "n_ties"])


# -- Per-method aggregation --------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """One aggregate row (`_global_` or `_macro_avg_`) collapsed to its
    headline triple. For `_macro_avg_`, ``n`` is the number of qualifying
    subsets K, not pair count — see ``compute_pairwise_metrics``."""

    value: float
    se: float
    n: int


@dataclass(frozen=True)
class MethodMetrics:
    """One method × dataset, split into per-subset rows + both aggregates."""

    method: Method
    per_subset: pl.DataFrame  # cols: subset, value, se, n_pairs, n_ties
    global_: Aggregate
    macro_avg: Aggregate


def _split(df: pl.DataFrame) -> tuple[pl.DataFrame, Aggregate, Aggregate]:
    per_sub = df.filter(~pl.col("subset").is_in([GLOBAL_SUBSET, MACRO_AVG_SUBSET]))
    g = df.filter(pl.col("subset") == GLOBAL_SUBSET)
    m = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert g.height == 1 and m.height == 1, (
        f"expected exactly one _global_ and one _macro_avg_ row, got "
        f"global={g.height}, macro_avg={m.height}"
    )
    return (
        per_sub,
        Aggregate(
            value=float(g[0, "value"]),
            se=float(g[0, "se"]),
            n=int(g[0, "n_pairs"]),
        ),
        Aggregate(
            value=float(m[0, "value"]),
            se=float(m[0, "se"]),
            n=int(m[0, "n_pairs"]),
        ),
    )


# Families with externally-sourced metrics that we tolerate as missing
# (separate pipelines that may not have produced output for this dataset yet);
# first-party families ("bolinas", "conservation") fail loud.
_SOFT_FAIL_FAMILIES: frozenset[str] = frozenset({"alphagenome", "gpn_star"})


def gather_metrics(dataset: str) -> list[MethodMetrics]:
    """Fetch + split metrics for every method registered for ``dataset``,
    in registry order. Missing parquets for external families
    (``alphagenome`` / ``gpn_star``) print a warning and skip; first-party
    families fail loud."""
    out: list[MethodMetrics] = []
    for method in methods_for_dataset(dataset):
        try:
            df = fetch_method_metrics(method, dataset)
        except Exception as exc:  # noqa: BLE001
            if method.family in _SOFT_FAIL_FAMILIES:
                print(
                    f"  ! {method.family} metrics missing for "
                    f"{method.id} ({dataset}): {exc}"
                )
                continue
            raise
        per, g, m = _split(df)
        out.append(MethodMetrics(method=method, per_subset=per, global_=g, macro_avg=m))
    return out


def sort_by_leading(metrics: list[MethodMetrics], dataset: str) -> list[MethodMetrics]:
    """Stable descending sort by the leading aggregate for this dataset."""
    axis = LEADING_AGGREGATE[dataset]
    if axis == "global":
        key = lambda mm: -mm.global_.value  # noqa: E731
    else:
        key = lambda mm: -mm.macro_avg.value  # noqa: E731
    return sorted(metrics, key=key)


# -- Markdown rendering -------------------------------------------------------


def fmt(value: float, se: float) -> str:
    return f"{value:.3f} ± {se:.3f}"


def _method_label(mm: MethodMetrics) -> str:
    """Inline label for the leaderboard ``method`` column.

    Convention from the legacy ``leaderboard_gen.py``:
      - bolinas / alphagenome → ``` `display` (description) ```
      - conservation / gpn_star → ``` `display` ``` (no parenthetical)
    """
    name = f"`{mm.method.display}`"
    if mm.method.family in {"bolinas", "alphagenome"} and mm.method.description:
        return f"{name} ({mm.method.description})"
    return name


def build_table(dataset: str) -> str:
    """Render the per-dataset markdown leaderboard table.

    Format and ordering are preserved byte-for-byte from the pre-refactor
    ``leaderboard_gen.py`` so issue-body diffs stay clean during the
    transition to the dashboard."""
    metrics = sort_by_leading(gather_metrics(dataset), dataset)
    if not metrics:
        return f"# {dataset}\n\nNo methods registered.\n"

    leading = LEADING_AGGREGATE[dataset]

    subset_n: dict[str, int] = {}
    for mm in metrics:
        for s, n in mm.per_subset.select(["subset", "n_pairs"]).iter_rows():
            subset_n[s] = max(subset_n.get(s, 0), int(n))
    subsets = [
        s
        for s, n in sorted(subset_n.items(), key=lambda kv: -kv[1])
        if n >= N_MIN and s in SUBSET_DISPLAY
    ]
    if not subsets:
        return f"# {dataset}\n\nNo subset has n_pairs ≥ {N_MIN}.\n"

    cell: dict[tuple[str, str], tuple[float, float]] = {}
    top_subset: dict[str, float] = {}
    for mm in metrics:
        label = _method_label(mm)
        for row in mm.per_subset.iter_rows(named=True):
            s = row["subset"]
            if s in subsets:
                v, se = float(row["value"]), float(row["se"])
                cell[(label, s)] = (v, se)
                top_subset[s] = max(top_subset.get(s, -1.0), v)

    top_global = max(mm.global_.value for mm in metrics)
    top_macro = max(mm.macro_avg.value for mm in metrics)

    global_n = metrics[0].global_.n
    macro_k = metrics[0].macro_avg.n

    global_header = f"Global<br>(n={global_n})"
    macro_header = f"Macro Avg<br>({macro_k} subsets)"
    aggregate_headers = (
        [macro_header, global_header]
        if leading == "macro"
        else [global_header, macro_header]
    )
    header_cols = aggregate_headers + [
        f"{SUBSET_DISPLAY[s]}<br>(n={subset_n[s]})" for s in subsets
    ]
    header = "| method | " + " | ".join(header_cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(header_cols)) + "|"
    lines = [header, sep]

    for mm in metrics:
        label = _method_label(mm)
        gv, gse = mm.global_.value, mm.global_.se
        mv, mse = mm.macro_avg.value, mm.macro_avg.se
        global_cell = f"**{fmt(gv, gse)}**" if gv >= top_global - 0.01 else fmt(gv, gse)
        macro_cell = f"**{fmt(mv, mse)}**" if mv >= top_macro - 0.01 else fmt(mv, mse)
        cells = (
            [macro_cell, global_cell]
            if leading == "macro"
            else [global_cell, macro_cell]
        )
        for s in subsets:
            if (label, s) not in cell:
                cells.append("—")
                continue
            v, se = cell[(label, s)]
            text = fmt(v, se)
            if v >= top_subset[s] - 0.01:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# -- Normalized "long" form for downstream consumers --------------------------
# Used by ``dashboard/src/data/leaderboard.parquet.py`` to build one tidy
# parquet per dataset that the Observable Framework site reads via DuckDB.


def normalized_rows(dataset: str) -> pl.DataFrame:
    """Long-form table of all metrics for one dataset.

    Columns:
      - ``method_id`` — Method.id (primary key)
      - ``method_display`` — Method.display
      - ``family`` — Method.family
      - ``subset`` — consequence subset name OR ``_global_`` / ``_macro_avg_``
      - ``value`` — PairwiseAccuracy
      - ``se`` — Wald binomial SE
      - ``n_pairs`` — pair count (or K = qualifying subsets for ``_macro_avg_``)
      - ``n_ties`` — tied-pair count
    """
    rows: list[dict] = []
    for mm in gather_metrics(dataset):
        for row in mm.per_subset.iter_rows(named=True):
            rows.append(
                {
                    "method_id": mm.method.id,
                    "method_display": mm.method.display,
                    "family": mm.method.family,
                    "subset": row["subset"],
                    "value": float(row["value"]),
                    "se": float(row["se"]),
                    "n_pairs": int(row["n_pairs"]),
                    "n_ties": int(row["n_ties"]),
                }
            )
        for subset, agg in (
            (GLOBAL_SUBSET, mm.global_),
            (MACRO_AVG_SUBSET, mm.macro_avg),
        ):
            rows.append(
                {
                    "method_id": mm.method.id,
                    "method_display": mm.method.display,
                    "family": mm.method.family,
                    "subset": subset,
                    "value": agg.value,
                    "se": agg.se,
                    "n_pairs": agg.n,
                    "n_ties": 0,
                }
            )
    return pl.DataFrame(rows)

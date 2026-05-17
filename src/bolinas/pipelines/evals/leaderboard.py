"""Aggregate matched-pair evaluation metrics from S3 into leaderboard rows.

Reads pre-computed PairwiseAccuracy parquets emitted by these snakemake
pipelines, then collapses them into one ``ModelMetrics`` object per registered
method × dataset:

  - ``snakemake/analysis/evals_v2/``  → one parquet per ``(model, dataset)``,
    filter by ``score_type`` + ``split``.
  - ``snakemake/conservation_eval/``  → one parquet per ``(dataset, split)``,
    filter by ``score_name`` (the track).
  - ``snakemake/alphagenome_eval/``   → one parquet per dataset, filter by
    ``score_type`` + ``split``.
  - ``snakemake/gpn_star_eval/``      → one parquet per dataset, filter by
    ``score_type`` + ``split`` + ``model``.

Model registry (display name, family, training metadata, etc.) lives in
``dashboard/models.yaml`` and is loaded via ``methods.load_models``. This
module is the data layer for both the dashboard (``dashboard/``) and the
legacy issue-body patching CLI (``snakemake/evals/scratch/leaderboard_gen.py``).
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from typing import Literal

import polars as pl

from bolinas.pipelines.evals.models import ALL_DATASETS, Model, models_for_dataset
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET

S3 = "s3://oa-bolinas"
SPLIT = "train"
N_MIN = 30

# Per-family scoring protocols. Each protocol maps a dataset → the parquet
# `score_type` column to filter on. The DEFAULT protocol is the one the
# issue-body CLI and legacy code paths use. The dashboard exposes the
# non-default protocols (where present) as per-family toggle options.
#
# All protocols here are expected to be in the precomputed metrics parquet
# on S3. Adding a new protocol means: (1) extend `metrics.smk` to compute
# pairwise PA for that score column, (2) re-run `compute_metrics`, (3) add
# the protocol entry here.
PROTOCOLS: dict[str, dict[str, dict[str, str]]] = {
    "bolinas": {
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
            "eqtl": "abs_llr",
        },
        "JSD": {d: "next_token_jsd_mean" for d in ALL_DATASETS},
    },
    "conservation": {
        "score": {d: "score" for d in ALL_DATASETS},
    },
    "alphagenome": {
        "L2": {d: "alphagenome_max_l2" for d in ALL_DATASETS},
    },
    "gpn_star": {
        "cLLR": {
            "mendelian_traits": "minus_llr_calibrated",
            "complex_traits": "abs_llr_calibrated",
            "eqtl": "abs_llr_calibrated",
        },
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
            "eqtl": "abs_llr",
        },
    },
}

DEFAULT_PROTOCOL: dict[str, str] = {
    "bolinas": "LLR",
    "conservation": "score",
    "alphagenome": "L2",
    "gpn_star": "cLLR",
}


def score_type_for(family: str, protocol: str, dataset: str) -> str:
    """Score-column name for one (family, protocol, dataset) combination."""
    return PROTOCOLS[family][protocol][dataset]


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


def _parquet_path(method: Model, dataset: str) -> str:
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


def fetch_method_metrics(
    method: Model, dataset: str, protocol: str | None = None
) -> pl.DataFrame:
    """Return rows ``[subset, value, se, n_pairs, n_ties]`` for one
    ``(method, dataset, protocol)`` — including the ``_global_`` and
    ``_macro_avg_`` aggregate rows.

    When ``protocol`` is ``None``, defaults to ``DEFAULT_PROTOCOL[family]``.
    """
    assert dataset in method.datasets, (
        f"{method.id!r} is not registered for dataset {dataset!r}"
    )
    protocol = protocol or DEFAULT_PROTOCOL[method.family]
    assert protocol in PROTOCOLS[method.family], (
        f"unknown protocol {protocol!r} for family {method.family!r}; "
        f"options: {list(PROTOCOLS[method.family])}"
    )
    score_type = PROTOCOLS[method.family][protocol][dataset]
    path = _parquet_path(method, dataset)
    df = _read_parquet(path)
    match method.family:
        case "bolinas" | "alphagenome":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("split") == SPLIT
            )
        case "conservation":
            df = df.filter(pl.col("score_name") == method.id)
        case "gpn_star":
            df = (
                df.filter(pl.col("score_type") == score_type)
                .filter(pl.col("split") == SPLIT)
                .filter(pl.col("model") == method.id)
            )
        case _:
            raise ValueError(f"unknown family {method.family!r}")
    if df.height == 0:
        raise LookupError(
            f"no metrics rows for {method.id!r} on {dataset!r} with protocol "
            f"{protocol!r} (score_type={score_type!r}) in {path}. The pipeline "
            f"may need to be re-run with this protocol included."
        )
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
class ModelMetrics:
    """One method × dataset, split into per-subset rows + both aggregates."""

    method: Model
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


def gather_metrics(
    dataset: str, protocols: dict[str, str] | None = None
) -> list[ModelMetrics]:
    """Fetch + split metrics for every method registered for ``dataset``,
    in registry order.

    ``protocols`` (family → protocol name) overrides the default protocol
    on a per-family basis; families not present in the dict use
    ``DEFAULT_PROTOCOL``. Missing parquets for external families
    (``alphagenome`` / ``gpn_star``) print a warning and skip; first-party
    families fail loud.
    """
    overrides = protocols or {}
    out: list[ModelMetrics] = []
    for method in models_for_dataset(dataset):
        protocol = overrides.get(method.family, DEFAULT_PROTOCOL[method.family])
        try:
            df = fetch_method_metrics(method, dataset, protocol)
        except Exception as exc:  # noqa: BLE001
            if method.family in _SOFT_FAIL_FAMILIES:
                print(
                    f"  ! {method.family} metrics missing for "
                    f"{method.id} ({dataset}, {protocol}): {exc}",
                    file=sys.stderr,
                )
                continue
            raise
        per, g, m = _split(df)
        out.append(ModelMetrics(method=method, per_subset=per, global_=g, macro_avg=m))
    return out


def sort_by_leading(metrics: list[ModelMetrics], dataset: str) -> list[ModelMetrics]:
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


def _model_label(mm: ModelMetrics) -> str:
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
        label = _model_label(mm)
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
        label = _model_label(mm)
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

    Emits one row per ``(method, protocol, subset)`` — so each method
    contributes one row block per protocol registered for its family.

    Protocols whose metrics aren't in the parquet yet (e.g. the JSD path
    before ``compute_metrics`` has been re-run with the new score column)
    log a warning and are skipped rather than failing the build.

    Columns:
      - ``method_id``       — ``Model.id`` (primary key for a row's method)
      - ``method_display``  — ``Model.display``
      - ``family``          — ``Model.family``
      - ``protocol``        — protocol name (e.g. ``LLR``, ``JSD``, ``cLLR``)
      - ``subset``          — consequence subset OR ``_global_`` / ``_macro_avg_``
      - ``value``           — PairwiseAccuracy
      - ``se``              — Wald binomial SE
      - ``n_pairs``         — pair count (or K = qualifying subsets for ``_macro_avg_``)
      - ``n_ties``          — tied-pair count
    """
    # Soft-fail surface, intentionally narrow: only the two legitimate
    # "no data for this protocol yet" exception types.
    #   * `LookupError` — `fetch_method_metrics` raises this when the
    #     parquet exists but has no rows for the requested protocol's
    #     `score_type` (e.g. bolinas JSD before `metrics.smk` was rerun
    #     with the new score column).
    #   * `pl.exceptions.ComputeError` — `pl.read_parquet` raises this
    #     when the parquet file isn't on S3 yet (e.g. a freshly added
    #     external-family entry that the eval pipeline hasn't produced
    #     output for). `FileNotFoundError` covers the local-path case in
    #     tests.
    # Everything else (`AssertionError`, `KeyError` from a malformed
    # registry, `ValueError`, etc.) propagates so config bugs fail loud
    # instead of yielding a silently-empty dashboard.
    soft_fail = (LookupError, pl.exceptions.ComputeError, FileNotFoundError)
    rows: list[dict] = []
    for method in models_for_dataset(dataset):
        for protocol in PROTOCOLS[method.family]:
            try:
                df = fetch_method_metrics(method, dataset, protocol)
            except soft_fail as exc:
                print(
                    f"  ! {method.family}/{protocol} skip for {method.id} "
                    f"({dataset}): {exc}",
                    file=sys.stderr,
                )
                continue
            for row in df.iter_rows(named=True):
                rows.append(
                    {
                        "method_id": method.id,
                        "method_display": method.display,
                        "family": method.family,
                        "protocol": protocol,
                        "subset": row["subset"],
                        "value": float(row["value"]),
                        "se": float(row["se"]),
                        "n_pairs": int(row["n_pairs"]),
                        "n_ties": int(row["n_ties"]),
                    }
                )
    return pl.DataFrame(rows)

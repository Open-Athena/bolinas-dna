"""Tests for ``bolinas.pipelines.evals.leaderboard``.

Most tests bypass the cached S3 reader by monkeypatching ``_read_parquet`` to
return synthetic DataFrames keyed off the parquet path. This lets us exercise
the full aggregation + rendering pipeline without network access.
"""

from __future__ import annotations


import polars as pl
import pytest

from bolinas.pipelines.evals import leaderboard
from bolinas.pipelines.evals.leaderboard import (
    Aggregate,
    LEADING_AGGREGATE,
    MethodMetrics,
    SUBSET_DISPLAY,
    _method_label,
    _split,
    build_table,
    fmt,
    normalized_rows,
    sort_by_leading,
)
from bolinas.pipelines.evals.methods import Method
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET


def _make_metrics_df(
    per_subset: dict[str, tuple[float, float, int, int]],
    global_: tuple[float, float, int, int],
    macro: tuple[float, float, int, int],
) -> pl.DataFrame:
    """Build a metrics DataFrame in the shape the library expects.

    Each value tuple is ``(value, se, n_pairs, n_ties)``.
    """
    rows = []
    for subset, (v, se, n, t) in per_subset.items():
        rows.append({"subset": subset, "value": v, "se": se, "n_pairs": n, "n_ties": t})
    for label, tup in ((GLOBAL_SUBSET, global_), (MACRO_AVG_SUBSET, macro)):
        v, se, n, t = tup
        rows.append({"subset": label, "value": v, "se": se, "n_pairs": n, "n_ties": t})
    return pl.DataFrame(rows)


# ---- _split -----------------------------------------------------------------


def test_split_extracts_aggregates_and_keeps_per_subset():
    df = _make_metrics_df(
        per_subset={
            "missense_variant": (0.7, 0.02, 100, 1),
            "splicing": (0.65, 0.05, 50, 0),
        },
        global_=(0.69, 0.018, 150, 1),
        macro=(0.675, 0.027, 2, 1),
    )
    per_sub, g, m = _split(df)
    assert per_sub.height == 2
    assert set(per_sub["subset"].to_list()) == {"missense_variant", "splicing"}
    assert g == Aggregate(value=0.69, se=0.018, n=150)
    assert m == Aggregate(value=0.675, se=0.027, n=2)


def test_split_fails_without_global():
    df = _make_metrics_df(
        per_subset={"missense_variant": (0.7, 0.02, 100, 1)},
        global_=(0.7, 0.02, 100, 1),
        macro=(0.7, 0.02, 1, 1),
    )
    df = df.filter(pl.col("subset") != GLOBAL_SUBSET)
    with pytest.raises(AssertionError, match="exactly one _global_"):
        _split(df)


# ---- _method_label ----------------------------------------------------------


def _mk_method(
    id: str = "x",
    display: str = "x",
    family: str = "bolinas",
    description: str = "desc",
    datasets: tuple[str, ...] = ("mendelian_traits",),
    **extra,
) -> Method:
    return Method(
        id=id,
        display=display,
        family=family,  # type: ignore[arg-type]
        description=description,
        datasets=datasets,
        checkpoint=extra.get("checkpoint"),
    )


def _mk_mm(method: Method) -> MethodMetrics:
    return MethodMetrics(
        method=method,
        per_subset=pl.DataFrame(
            {"subset": [], "value": [], "se": [], "n_pairs": [], "n_ties": []}
        ),
        global_=Aggregate(0.0, 0.0, 0),
        macro_avg=Aggregate(0.0, 0.0, 0),
    )


def test_method_label_bolinas_includes_description():
    mm = _mk_mm(
        _mk_method(
            display="exp55-mammals", family="bolinas", description="promoters, mammals"
        )
    )
    assert _method_label(mm) == "`exp55-mammals` (promoters, mammals)"


def test_method_label_alphagenome_includes_description():
    mm = _mk_mm(
        _mk_method(
            display="AlphaGenome",
            family="alphagenome",
            description="variant scorer, API",
        )
    )
    assert _method_label(mm) == "`AlphaGenome` (variant scorer, API)"


def test_method_label_conservation_excludes_description():
    mm = _mk_mm(
        _mk_method(
            display="phyloP_241m",
            family="conservation",
            description="phyloP across 241 mammals",
        )
    )
    assert _method_label(mm) == "`phyloP_241m`"


def test_method_label_gpn_star_excludes_description():
    mm = _mk_mm(
        _mk_method(
            display="GPN-Star-M", family="gpn_star", description="mammal, 447-way MSA"
        )
    )
    assert _method_label(mm) == "`GPN-Star-M`"


# ---- sort_by_leading --------------------------------------------------------


def _mk_mm_with_aggs(
    display: str, family: str, global_value: float, macro_value: float
) -> MethodMetrics:
    return MethodMetrics(
        method=_mk_method(id=display, display=display, family=family),
        per_subset=pl.DataFrame(
            {"subset": [], "value": [], "se": [], "n_pairs": [], "n_ties": []}
        ),
        global_=Aggregate(global_value, 0.0, 1000),
        macro_avg=Aggregate(macro_value, 0.0, 5),
    )


def test_sort_by_leading_uses_macro_for_mendelian():
    a = _mk_mm_with_aggs("a", "bolinas", global_value=0.9, macro_value=0.5)
    b = _mk_mm_with_aggs("b", "bolinas", global_value=0.6, macro_value=0.8)
    out = sort_by_leading([a, b], "mendelian_traits")
    assert [mm.method.display for mm in out] == ["b", "a"]


def test_sort_by_leading_uses_global_for_complex():
    a = _mk_mm_with_aggs("a", "bolinas", global_value=0.9, macro_value=0.5)
    b = _mk_mm_with_aggs("b", "bolinas", global_value=0.6, macro_value=0.8)
    out = sort_by_leading([a, b], "complex_traits")
    assert [mm.method.display for mm in out] == ["a", "b"]


def test_sort_is_stable_on_ties():
    a = _mk_mm_with_aggs("a", "bolinas", global_value=0.7, macro_value=0.5)
    b = _mk_mm_with_aggs("b", "bolinas", global_value=0.7, macro_value=0.5)
    out = sort_by_leading([a, b], "complex_traits")
    assert [mm.method.display for mm in out] == ["a", "b"]


# ---- fmt --------------------------------------------------------------------


def test_fmt_rounds_to_three_decimals():
    assert fmt(0.679, 0.018) == "0.679 ± 0.018"
    assert fmt(0.6789, 0.018) == "0.679 ± 0.018"


# ---- subset display mapping is exhaustive on the actual leaderboard subsets


def test_subset_display_covers_all_keys():
    # Sanity: each key has a value that's non-empty
    assert all(SUBSET_DISPLAY.values())


# ---- end-to-end build_table with synthetic S3 reads -------------------------


def _patch_read_parquet(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, pl.DataFrame],
) -> None:
    """Monkeypatch ``_read_parquet`` to return synthetic data keyed by path.

    We also flush the existing lru_cache so cached real-S3 results from prior
    tests don't leak through.
    """
    leaderboard._read_parquet.cache_clear()

    def fake(path: str) -> pl.DataFrame:
        if path not in responses:
            raise FileNotFoundError(f"no synthetic response for {path!r}")
        return responses[path]

    monkeypatch.setattr(leaderboard, "_read_parquet", fake)


def _patch_methods(
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple[Method, ...],
) -> None:
    """Bypass the real methods.yaml so build_table operates on a small fixture."""
    monkeypatch.setattr(
        "bolinas.pipelines.evals.methods.load_methods",
        lambda: methods,
    )


def test_build_table_minimal_two_method_one_subset(monkeypatch: pytest.MonkeyPatch):
    methods = (
        _mk_method(
            id="phyloP_241m",
            display="phyloP_241m",
            family="conservation",
            description="phyloP",
            datasets=("mendelian_traits",),
        ),
        _mk_method(
            id="exp55-mammals",
            display="exp55-mammals",
            family="bolinas",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)

    # Conservation has one shared parquet per dataset, with score_name filtering.
    cons_df = pl.DataFrame(
        [
            {
                "score_name": "phyloP_241m",
                "subset": "missense_variant",
                "value": 0.75,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 1,
            },
            {
                "score_name": "phyloP_241m",
                "subset": GLOBAL_SUBSET,
                "value": 0.74,
                "se": 0.018,
                "n_pairs": 150,
                "n_ties": 1,
            },
            {
                "score_name": "phyloP_241m",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.75,
                "se": 0.025,
                "n_pairs": 1,
                "n_ties": 1,
            },
        ]
    )
    bolinas_df = pl.DataFrame(
        [
            {
                "score_type": "minus_llr",
                "split": "train",
                "subset": "missense_variant",
                "value": 0.80,
                "se": 0.015,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_type": "minus_llr",
                "split": "train",
                "subset": GLOBAL_SUBSET,
                "value": 0.79,
                "se": 0.014,
                "n_pairs": 150,
                "n_ties": 0,
            },
            {
                "score_type": "minus_llr",
                "split": "train",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.80,
                "se": 0.022,
                "n_pairs": 1,
                "n_ties": 0,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "mendelian_traits/metrics_train.parquet": cons_df,
            "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
            "exp55-mammals/mendelian_traits.parquet": bolinas_df,
        },
    )

    table = build_table("mendelian_traits")
    # Mendelian sorts by Macro Avg descending → exp55-mammals (0.80) above phyloP_241m (0.75)
    lines = table.split("\n")
    assert lines[0].startswith("| method |")
    assert "Macro Avg" in lines[0]  # leading aggregate is macro for mendelian
    assert lines[0].index("Macro Avg") < lines[0].index("Global")
    # exp55 should come before phyloP
    method_col = [line.split("|")[1].strip() for line in lines[2:]]
    assert method_col[0].startswith("`exp55-mammals`")
    assert method_col[1].startswith("`phyloP_241m`")


def test_build_table_complex_uses_global_axis(monkeypatch: pytest.MonkeyPatch):
    methods = (
        _mk_method(
            id="phyloP_241m",
            display="phyloP_241m",
            family="conservation",
            description="phyloP",
            datasets=("complex_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    cons_df = pl.DataFrame(
        [
            {
                "score_name": "phyloP_241m",
                "subset": "distal",
                "value": 0.6,
                "se": 0.02,
                "n_pairs": 400,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_241m",
                "subset": GLOBAL_SUBSET,
                "value": 0.55,
                "se": 0.02,
                "n_pairs": 500,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_241m",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.6,
                "se": 0.02,
                "n_pairs": 1,
                "n_ties": 0,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "complex_traits/metrics_train.parquet": cons_df,
        },
    )
    table = build_table("complex_traits")
    header = table.split("\n")[0]
    # Complex uses Global leading
    assert header.index("Global") < header.index("Macro Avg")


def test_build_table_bolds_top_within_tolerance(monkeypatch: pytest.MonkeyPatch):
    """Top method + any within 0.01 PA of the top get bolded."""
    methods = (
        _mk_method(
            id="phyloP_top",
            display="phyloP_top",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
        _mk_method(
            id="phyloP_close",
            display="phyloP_close",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
        _mk_method(
            id="phyloP_far",
            display="phyloP_far",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    cons_df = pl.DataFrame(
        [
            # top (0.800 macro)
            {
                "score_name": "phyloP_top",
                "subset": "missense_variant",
                "value": 0.80,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_top",
                "subset": GLOBAL_SUBSET,
                "value": 0.80,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_top",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.80,
                "se": 0.02,
                "n_pairs": 1,
                "n_ties": 0,
            },
            # within 0.01 of top (0.795)
            {
                "score_name": "phyloP_close",
                "subset": "missense_variant",
                "value": 0.795,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_close",
                "subset": GLOBAL_SUBSET,
                "value": 0.795,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_close",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.795,
                "se": 0.02,
                "n_pairs": 1,
                "n_ties": 0,
            },
            # far below (0.70)
            {
                "score_name": "phyloP_far",
                "subset": "missense_variant",
                "value": 0.70,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_far",
                "subset": GLOBAL_SUBSET,
                "value": 0.70,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_far",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.70,
                "se": 0.02,
                "n_pairs": 1,
                "n_ties": 0,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "mendelian_traits/metrics_train.parquet": cons_df,
        },
    )
    table = build_table("mendelian_traits")
    # phyloP_top and phyloP_close should be bolded in the Macro Avg column (top - 0.01 tolerance).
    # phyloP_far should NOT be bolded.
    lines = table.split("\n")
    top_line = next(line for line in lines if "phyloP_top" in line)
    close_line = next(line for line in lines if "phyloP_close" in line)
    far_line = next(line for line in lines if "phyloP_far" in line)
    assert "**0.800 ± 0.020**" in top_line
    assert "**0.795 ± 0.020**" in close_line
    assert "**0.700 ± 0.020**" not in far_line


def test_alphagenome_missing_metrics_soft_fails(monkeypatch: pytest.MonkeyPatch):
    """A missing alphagenome parquet should log + skip, not crash the table."""
    methods = (
        _mk_method(
            id="phyloP_241m",
            display="phyloP_241m",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
        _mk_method(
            id="AlphaGenome",
            display="AlphaGenome",
            family="alphagenome",
            description="variant scorer, API",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    cons_df = pl.DataFrame(
        [
            {
                "score_name": "phyloP_241m",
                "subset": "missense_variant",
                "value": 0.7,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_241m",
                "subset": GLOBAL_SUBSET,
                "value": 0.7,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_241m",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.7,
                "se": 0.02,
                "n_pairs": 1,
                "n_ties": 0,
            },
        ]
    )
    # alphagenome parquet intentionally absent from responses dict.
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "mendelian_traits/metrics_train.parquet": cons_df,
        },
    )
    table = build_table("mendelian_traits")
    # AlphaGenome row should be absent; conservation row present.
    assert "AlphaGenome" not in table
    assert "phyloP_241m" in table


def test_bolinas_missing_metrics_hard_fails(monkeypatch: pytest.MonkeyPatch):
    methods = (
        _mk_method(
            id="exp55-mammals",
            display="exp55-mammals",
            family="bolinas",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    _patch_read_parquet(monkeypatch, {})  # no responses → FileNotFoundError on read
    with pytest.raises(FileNotFoundError):
        build_table("mendelian_traits")


def test_leading_aggregate_constants_cover_all_datasets():
    from bolinas.pipelines.evals.methods import ALL_DATASETS

    assert set(LEADING_AGGREGATE.keys()) == set(ALL_DATASETS)


# ---- normalized_rows --------------------------------------------------------


def test_normalized_rows_includes_aggregates_and_per_subset(
    monkeypatch: pytest.MonkeyPatch,
):
    methods = (
        _mk_method(
            id="phyloP_241m",
            display="phyloP_241m",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    cons_df = pl.DataFrame(
        [
            {
                "score_name": "phyloP_241m",
                "subset": "missense_variant",
                "value": 0.75,
                "se": 0.02,
                "n_pairs": 100,
                "n_ties": 1,
            },
            {
                "score_name": "phyloP_241m",
                "subset": "splicing",
                "value": 0.65,
                "se": 0.05,
                "n_pairs": 40,
                "n_ties": 0,
            },
            {
                "score_name": "phyloP_241m",
                "subset": GLOBAL_SUBSET,
                "value": 0.72,
                "se": 0.018,
                "n_pairs": 140,
                "n_ties": 1,
            },
            {
                "score_name": "phyloP_241m",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.70,
                "se": 0.025,
                "n_pairs": 2,
                "n_ties": 1,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "mendelian_traits/metrics_train.parquet": cons_df,
        },
    )
    df = normalized_rows("mendelian_traits")
    assert set(df["subset"].to_list()) == {
        "missense_variant",
        "splicing",
        GLOBAL_SUBSET,
        MACRO_AVG_SUBSET,
    }
    assert set(df.columns) == {
        "method_id",
        "method_display",
        "family",
        "subset",
        "value",
        "se",
        "n_pairs",
        "n_ties",
    }

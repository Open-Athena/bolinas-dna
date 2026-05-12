"""Generate leaderboard markdown tables for issues #161 / #162 / #172.

Pulls per-(method, dataset, subset) PairwiseAccuracy + SE from S3:
  - conservation_eval: 7 conservation tracks
  - evals_v2: 5 model checkpoints
  - alphagenome_eval: AlphaGenome variant scorer

Combines into one table per dataset. n_pairs ≥ 30 cutoff for per-subset
columns. Two aggregate columns are prepended:

  - Global: PA across ALL pairs (no n filter). Sourced from the `_global_`
    row written by `compute_pairwise_metrics`.
  - Macro Avg: unweighted mean of per-subset PAs over n≥30 subsets. Sourced
    from the `_macro_avg_` row.

Bolding rule (top method per column, plus any within 0.01 of the top) applies
to every column including the aggregates.

Outputs three markdown chunks ready to drop into the leaderboard issues. With
`--patch-issues`, surgically PATCHes the bodies of #161/#162/#172 instead of
relying on manual paste.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date

import polars as pl

from bolinas.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET

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
DATASET_ISSUE = {
    "mendelian_traits": 161,
    "complex_traits": 162,
    "eqtl": 172,
}
S3 = "s3://oa-bolinas"
SPLIT = "train"
N_MIN = 30
REPO = "Open-Athena/bolinas-dna"


def fmt(value: float, se: float) -> str:
    return f"{value:.3f} ± {se:.3f}"


def gather_methods(dataset: str) -> list[tuple[str, str | None, pl.DataFrame]]:
    """Return [(method_name, comment, full_df), ...] for one dataset.

    full_df includes both per-subset rows and the aggregate `_global_` /
    `_macro_avg_` rows."""
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

    # Per-method aggregate rows (_global_ + _macro_avg_).
    agg: dict[str, dict[str, tuple[float, float, int]]] = {}
    for method, _, df in rows:
        g = df.filter(pl.col("subset") == GLOBAL_SUBSET)
        m = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
        assert g.height == 1, f"{method}: expected 1 _global_ row, got {g.height}"
        assert m.height == 1, f"{method}: expected 1 _macro_avg_ row, got {m.height}"
        agg[method] = {
            "global": (g[0, "value"], g[0, "se"], int(g[0, "n_pairs"])),
            "macro": (m[0, "value"], m[0, "se"], int(m[0, "n_pairs"])),
        }

    # Per-subset coverage (excluding aggregate rows).
    subset_n: dict[str, int] = {}
    for _, _, df in rows:
        per_sub = df.filter(~pl.col("subset").is_in([GLOBAL_SUBSET, MACRO_AVG_SUBSET]))
        for s, n in per_sub.select(["subset", "n_pairs"]).iter_rows():
            subset_n[s] = max(subset_n.get(s, 0), int(n))
    subsets = [
        s
        for s, n in sorted(subset_n.items(), key=lambda kv: -kv[1])
        if n >= N_MIN and s in SUBSET_DISPLAY
    ]
    if not subsets:
        return f"# {dataset}\n\nNo subset has n_pairs ≥ {N_MIN}.\n"

    # Per-subset cell values + per-column top (for bolding).
    cell_pa: dict[tuple[str, str], tuple[float, float]] = {}
    top_subset: dict[str, float] = {}
    for method, _, df in rows:
        per_sub = df.filter(~pl.col("subset").is_in([GLOBAL_SUBSET, MACRO_AVG_SUBSET]))
        for s, v, se, _ in per_sub.iter_rows():
            if s in subsets:
                cell_pa[(method, s)] = (v, se)
                top_subset[s] = max(top_subset.get(s, -1.0), v)

    # Per-column tops for the two aggregate columns.
    top_global = max(agg[m]["global"][0] for m in agg)
    top_macro = max(agg[m]["macro"][0] for m in agg)

    # Aggregate column headers — these counts are the same across methods.
    any_method = next(iter(agg))
    global_n = agg[any_method]["global"][2]
    macro_k = agg[any_method]["macro"][2]

    header_cols = [
        f"Global<br>(n={global_n})",
        f"Macro Avg<br>({macro_k} subsets)",
    ] + [f"{SUBSET_DISPLAY[s]}<br>(n={subset_n[s]})" for s in subsets]
    header = "| method | " + " | ".join(header_cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(header_cols)) + "|"
    lines = [header, sep]

    for method, comment, _ in rows:
        label = method + (f" ({comment})" if comment else "")
        gv, gse, _ = agg[method]["global"]
        mv, mse, _ = agg[method]["macro"]
        cells = [
            f"**{fmt(gv, gse)}**" if gv >= top_global - 0.01 else fmt(gv, gse),
            f"**{fmt(mv, mse)}**" if mv >= top_macro - 0.01 else fmt(mv, mse),
        ]
        for s in subsets:
            if (method, s) not in cell_pa:
                cells.append("—")
                continue
            v, se = cell_pa[(method, s)]
            text = fmt(v, se)
            if v >= top_subset[s] - 0.01:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---- Issue body PATCH ------------------------------------------------------


def _gh_get_issue_body(issue_number: int) -> str:
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/{issue_number}", "--jq", ".body"],
        check=True,
        capture_output=True,
        text=True,
    )
    # The jq filter strips the surrounding JSON quoting but the trailing
    # newline from subprocess capture is artifactual — issue bodies do not
    # end with a newline themselves.
    return out.stdout.rstrip("\n")


def _gh_patch_issue_body(issue_number: int, body: str) -> None:
    """PATCH /repos/.../issues/<n> via gh api with JSON stdin so multi-line
    bodies aren't subject to shell-quoting or form-encoding fragility."""
    subprocess.run(
        [
            "gh", "api", "-X", "PATCH",
            f"repos/{REPO}/issues/{issue_number}",
            "--input", "-",
        ],
        input=json.dumps({"body": body}),
        check=True,
        capture_output=True,
        text=True,
    )


# The leaderboard table is the only contiguous block of pipe-delimited lines
# anchored on "| method |" in the body. Other tables in <details> (Sizes,
# Reproducibility) start with "| subset" or "| method | step" and won't match.
TABLE_RE = re.compile(
    r"\| method \|[^\n]*\n(?:\|[^\n]*\n)+",
    re.MULTILINE,
)


def _replace_table(body: str, new_table: str) -> str:
    if not TABLE_RE.search(body):
        raise RuntimeError("could not find leaderboard table in issue body")
    return TABLE_RE.sub(new_table + "\n", body, count=1)


def _update_intro_paragraph(body: str, global_n: int, macro_k: int) -> str:
    """Update the paragraph just before the table to explain the two new
    aggregate columns. Anchored on the exact pre-edit phrasing; idempotent
    because the replacement no longer matches."""
    new_intro = (
        f"PairwiseAccuracy ± SE per method. Higher is better. Two leftmost "
        f"columns aggregate across the per-subset cells: **Global** = PA "
        f"across **all** matched pairs (n={global_n}), including any "
        f"subsets below the threshold; **Macro Avg** = unweighted mean of "
        f"per-subset PAs over the {macro_k} reported subsets. Subsets with "
        f"`n_pairs < 30` are excluded from the per-subset columns by "
        f"convention (held across leaderboard datasets in this repo)."
    )
    old_variants = [
        # Mendelian / eqtl phrasing.
        "PairwiseAccuracy ± SE per consequence subset. Higher is better. "
        "Subsets with `n_pairs < 30` are excluded by convention (held across "
        "leaderboard datasets in this repo).",
        # Complex traits phrasing (extra "Most consequence subsets…" sentence).
        "PairwiseAccuracy ± SE per consequence subset. Higher is better. "
        "Subsets with `n_pairs < 30` are excluded by convention (held across "
        "leaderboard datasets in this repo). Most consequence subsets in "
        "this dataset fall below that threshold — see Sizes for full breakdown.",
    ]
    for old in old_variants:
        if old in body:
            return body.replace(old, new_intro, 1)
    # Already updated on a prior PATCH — idempotent path.
    if new_intro in body:
        return body
    raise RuntimeError("could not locate intro paragraph for update")


def _update_bold_rule(body: str) -> str:
    """Generalize "per subset" → "per column" since aggregates are columns
    too. Idempotent."""
    old = (
        "**Bold** = top method per subset, plus any method within **0.01** "
        "PairwiseAccuracy of the top."
    )
    new = (
        "**Bold** = top method per column (including aggregates), plus any "
        "method within **0.01** PairwiseAccuracy of the top."
    )
    if old in body:
        return body.replace(old, new, 1)
    if new in body:
        return body
    raise RuntimeError("could not locate bold-rule sentence for update")


def _bump_last_updated(body: str, today: str) -> str:
    return re.sub(
        r"\*\*Last updated:\*\* [0-9]{4}-[0-9]{2}-[0-9]{2}",
        f"**Last updated:** {today}",
        body,
        count=1,
    )


def _prepend_changelog_entry(body: str, entry: str) -> str:
    """Insert a new changelog item at the top of the Changelog <details>
    block. Idempotent: if the entry text already appears verbatim, no-op."""
    anchor = (
        "Protocol or data changes go below. The body above always reflects "
        "the *current* protocol; this section is the audit trail."
    )
    if anchor not in body:
        raise RuntimeError("missing changelog anchor")
    if entry in body:
        return body
    return body.replace(
        anchor + "\n\n",
        anchor + "\n\n" + entry + "\n\n",
        1,
    )


def patch_issue(dataset: str, table_md: str) -> None:
    issue_number = DATASET_ISSUE[dataset]
    print(f"  ↳ fetching issue #{issue_number} body...")
    body = _gh_get_issue_body(issue_number)

    # Pull global_n / macro_k out of the rendered table for the explainer.
    m_global = re.search(r"Global<br>\(n=(\d+)\)", table_md)
    m_macro = re.search(r"Macro Avg<br>\((\d+) subsets\)", table_md)
    assert m_global and m_macro, "could not parse aggregate-column counts"
    global_n = int(m_global.group(1))
    macro_k = int(m_macro.group(1))

    new_body = _replace_table(body, table_md)
    new_body = _update_intro_paragraph(new_body, global_n, macro_k)
    new_body = _update_bold_rule(new_body)
    new_body = _bump_last_updated(new_body, str(date.today()))
    new_body = _prepend_changelog_entry(
        new_body,
        f"- **{date.today()}** — added `Global` (PA across all pairs, "
        f"n={global_n}) and `Macro Avg` (unweighted PA over n≥30 subsets, "
        f"{macro_k} subsets) aggregate columns. Implemented as new "
        f"`_global_` / `_macro_avg_` rows in `compute_pairwise_metrics` "
        f"(`src/bolinas/evals/metrics.py`); all 3 metric pipelines re-run.",
    )

    if new_body == body:
        print(f"  ↳ no changes for issue #{issue_number}")
        return
    print(f"  ↳ patching issue #{issue_number}...")
    _gh_patch_issue_body(issue_number, new_body)
    print(f"  ↳ patched #{issue_number}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-issues",
        action="store_true",
        help="After printing, surgically PATCH the bodies of #161/#162/#172.",
    )
    args = parser.parse_args()

    tables: dict[str, str] = {}
    for ds in DATASETS:
        print(f"\n{'#' * 70}\n# {ds}\n{'#' * 70}\n")
        table = build_table(ds)
        print(table)
        tables[ds] = table

    if args.patch_issues:
        print("\n" + "#" * 70)
        print("# Patching GitHub issues")
        print("#" * 70)
        for ds in DATASETS:
            patch_issue(ds, tables[ds])


if __name__ == "__main__":
    main()

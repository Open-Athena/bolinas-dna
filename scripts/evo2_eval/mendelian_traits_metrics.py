"""Aggregate per-model Evo2 PairwiseAccuracy parquets into a single table
ready to merge into issue #161's Mendelian leaderboard.

Discovers ``{model}_{split}_metrics.parquet`` under
``results/evo2_mendelian_traits/``, concatenates, and writes:

- ``results/evo2_mendelian_traits/metrics.parquet`` — concatenated metrics.
- ``results/evo2_mendelian_traits/results_table.md`` — markdown rows in the
  leaderboard's column order (Global, Macro Avg, Missense, 5' UTR, ...).

The markdown is intended to be hand-merged into the issue body
(per project convention: leaderboards live in the issue body, not in
comments). Bolding (top method per column) is left to the human paste —
we only render our own Evo2 rows here.
"""

import argparse
from pathlib import Path

import pandas as pd

from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET


# Display name per subset (no hardcoded `n`s — the actual `n_pairs` are
# read from the metrics parquet so the table reflects the input data;
# works for both mendelian and complex_traits without per-dataset
# constants). Display order is mendelian's leaderboard order; subsets
# not in this list (or missing from the parquet) are rendered after,
# sorted by n_pairs descending.
SUBSET_DISPLAY: list[tuple[str, str]] = [
    ("missense_variant", "Missense"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("splicing", "Splicing"),
    ("tss_proximal", "Promoter"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("distal", "Distal"),
    ("non_coding_transcript_exon_variant", "ncRNA"),
    ("synonymous_variant", "Synonymous"),
]


def _fmt_cell(value: float, se: float) -> str:
    return f"{value:.3f} ± {se:.3f}"


def render_markdown(metrics: pd.DataFrame, score_column: str = "minus_llr") -> str:
    """Render the metrics DataFrame as a markdown table mirroring issue #161.

    Args:
        metrics: Concatenated metrics DataFrame with columns
            ``[score_type, subset, value, se, n_pairs, n_ties, model, ...]``.
        score_column: Score column to render (default ``minus_llr``).

    Returns:
        Markdown string (no trailing newline).
    """
    m = metrics[metrics["score_type"] == score_column].copy()
    assert not m.empty, f"no rows with score_type={score_column!r}"

    # Per-subset n_pairs (taken from any row — same value across models).
    n_pairs_by_subset: dict[str, int] = {
        s: int(m[m["subset"] == s]["n_pairs"].iloc[0]) for s in m["subset"].unique()
    }
    # Subset display order: SUBSET_DISPLAY entries first (in that order) if
    # present; remaining subsets after, sorted by n_pairs desc.
    known = [name for name, _ in SUBSET_DISPLAY]
    extras = sorted(
        (
            s
            for s in n_pairs_by_subset
            if s not in known and s not in (GLOBAL_SUBSET, MACRO_AVG_SUBSET)
        ),
        key=lambda s: -n_pairs_by_subset[s],
    )
    display: list[tuple[str, str]] = [
        (name, label) for (name, label) in SUBSET_DISPLAY if name in n_pairs_by_subset
    ] + [(s, s) for s in extras]

    # One row per model.
    models = m["model"].drop_duplicates().tolist()
    rows: list[tuple[str, float, dict[str, str]]] = []
    for model in models:
        sub = m[m["model"] == model].set_index("subset")
        if GLOBAL_SUBSET not in sub.index:
            raise AssertionError(f"model {model!r} missing {GLOBAL_SUBSET} row")
        if MACRO_AVG_SUBSET not in sub.index:
            raise AssertionError(f"model {model!r} missing {MACRO_AVG_SUBSET} row")
        glb = sub.loc[GLOBAL_SUBSET]
        mac = sub.loc[MACRO_AVG_SUBSET]
        cells = {
            "Global": _fmt_cell(glb["value"], glb["se"]),
            "Macro Avg": _fmt_cell(mac["value"], mac["se"]),
        }
        for subset_name, label in display:
            if subset_name in sub.index:
                row = sub.loc[subset_name]
                cells[label] = _fmt_cell(row["value"], row["se"])
            else:
                cells[label] = "—"
        rows.append((model, float(glb["value"]), cells))

    # Sort by Global PA descending (matches leaderboard convention).
    rows.sort(key=lambda r: r[1], reverse=True)

    global_n = n_pairs_by_subset.get(GLOBAL_SUBSET, 0)
    macro_k = n_pairs_by_subset.get(MACRO_AVG_SUBSET, 0)
    headers = [
        "method",
        f"Global<br>(n={global_n})",
        f"Macro Avg<br>({macro_k} subsets)",
    ]
    for subset_name, label in display:
        headers.append(f"{label}<br>(n={n_pairs_by_subset[subset_name]})")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for model, _, cells in rows:
        row_cells = [f"`{model}`", cells["Global"], cells["Macro Avg"]]
        for _, label in display:
            row_cells.append(cells[label])
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default="results/evo2_mendelian_traits",
        help="Directory containing per-model metrics parquets.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split to aggregate (matches the {model}_{split}_metrics.parquet glob).",
    )
    p.add_argument(
        "--score-column",
        default="minus_llr",
        help="Score column to render (defaults to leaderboard's minus_llr).",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    assert results_dir.exists(), f"{results_dir} not found"
    pattern = f"*_{args.split}_metrics.parquet"
    parquets = sorted(results_dir.glob(pattern))
    assert parquets, f"no metrics parquets matching {pattern} in {results_dir}"

    print(f"[aggregator] {len(parquets)} parquet(s) under {results_dir}:")
    for p_ in parquets:
        print(f"  - {p_.name}")

    metrics = pd.concat([pd.read_parquet(p_) for p_ in parquets], ignore_index=True)
    metrics_path = results_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"[aggregator] wrote {metrics_path}  ({len(metrics)} rows)")

    md = render_markdown(metrics, score_column=args.score_column)
    md_path = results_dir / "results_table.md"
    md_path.write_text(md + "\n")
    print(f"[aggregator] wrote {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()

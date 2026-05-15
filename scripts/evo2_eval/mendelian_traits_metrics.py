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


# Display order + display name for each subset, matching issue #161's table.
# (subset, label, expected_n) — expected_n is informational only and reflects
# the train-split sizes documented in the issue body at the time of writing.
SUBSET_DISPLAY: list[tuple[str, str, int]] = [
    ("missense_variant", "Missense", 4495),
    ("5_prime_UTR_variant", "5' UTR", 87),
    ("splicing", "Splicing", 78),
    ("tss_proximal", "Promoter", 61),
    ("3_prime_UTR_variant", "3' UTR", 58),
    ("distal", "Distal", 56),
    ("non_coding_transcript_exon_variant", "ncRNA", 42),
    ("synonymous_variant", "Synonymous", 33),
]
GLOBAL_EXPECTED_N = 4910
MACRO_EXPECTED_SUBSETS = 8


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
        for subset_name, label, _ in SUBSET_DISPLAY:
            if subset_name in sub.index:
                row = sub.loc[subset_name]
                cells[label] = _fmt_cell(row["value"], row["se"])
            else:
                cells[label] = "—"
        rows.append((model, float(glb["value"]), cells))

    # Sort by Global PA descending (matches leaderboard convention).
    rows.sort(key=lambda r: r[1], reverse=True)

    headers = [
        "method",
        f"Global<br>(n={GLOBAL_EXPECTED_N})",
        f"Macro Avg<br>({MACRO_EXPECTED_SUBSETS} subsets)",
    ]
    for subset_name, label, expected_n in SUBSET_DISPLAY:
        headers.append(f"{label}<br>(n={expected_n})")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for model, _, cells in rows:
        row_cells = [f"`{model}`", cells["Global"], cells["Macro Avg"]]
        for _, label, _ in SUBSET_DISPLAY:
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

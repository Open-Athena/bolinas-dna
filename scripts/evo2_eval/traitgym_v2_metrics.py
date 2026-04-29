"""Aggregate Evo2 TraitGym v2 predictions into a single metrics table + markdown.

Reads predictions parquets written by scripts/evo2_eval/traitgym_v2.py, reuses
src/bolinas/evals/metrics.py:compute_metrics (same function as the rest of
the pipeline), and emits:

  results/evo2_traitgym_v2/metrics.parquet
  results/evo2_traitgym_v2/results_table.md

Intended for the GitHub comment on issue #131.
"""

import argparse
from pathlib import Path

import pandas as pd

from bolinas.evals.metrics import compute_metrics


# Canonical display order for known models. Parquets for any other models
# are still picked up by auto-discovery and appended after these (sorted) so
# new models surface in the table without code changes.
MODELS = [
    "evo2_1b_base",
    "evo2_7b",
    "evo2_7b_base",
    "evo2_20b",
    "evo2_40b",
    "evo2_40b_base",
]
SCORE_TYPES = ["minus_llr", "abs_llr"]
REQUIRED_COLS = ["chrom", "pos", "ref", "alt", "label", "subset", "minus_llr", "abs_llr"]


def _metrics_for_model(predictions_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(predictions_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, (
        f"{predictions_path} missing required columns: {missing}"
    )
    dataset = df[["chrom", "pos", "ref", "alt", "label", "subset"]].copy()
    scores = df[["minus_llr", "abs_llr"]].copy()
    metrics = compute_metrics(
        dataset=dataset,
        scores=scores,
        metrics=["AUPRC"],
        score_columns=SCORE_TYPES,
    )
    metrics["model"] = model_name
    return metrics


def _build_markdown(metrics: pd.DataFrame, model_order: list[str]) -> str:
    """Pivot metrics to one row per subset, one column per model (minus_llr AUPRC)."""
    primary = metrics[metrics["score_type"] == "minus_llr"].copy()

    # n_pos/n_neg is subset-only, take the first occurrence per subset
    coverage = (
        primary[["subset", "n_pos", "n_neg"]]
        .drop_duplicates(subset="subset")
        .set_index("subset")
    )

    pivot = primary.pivot_table(
        index="subset",
        columns="model",
        values="value",
        aggfunc="first",
    )
    # Ordering: global first, then subsets sorted by n_pos descending
    order = ["global"] + [s for s in coverage.sort_values("n_pos", ascending=False).index if s != "global"]
    pivot = pivot.reindex(order)
    cols = [m for m in model_order if m in pivot.columns]
    pivot = pivot[cols]

    lines = ["### TraitGym Mendelian v2 — AUPRC (minus_llr), train split", ""]
    header_cells = ["subset", "n_pos", "n_neg"] + cols
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for subset in pivot.index:
        if subset in coverage.index:
            n_pos = int(coverage.loc[subset, "n_pos"])
            n_neg = int(coverage.loc[subset, "n_neg"])
        else:
            n_pos = n_neg = 0
        vals = [
            f"{pivot.loc[subset, m]:.3f}" if pd.notna(pivot.loc[subset, m]) else "—"
            for m in cols
        ]
        lines.append("| " + " | ".join([subset, str(n_pos), str(n_neg)] + vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions-dir",
        default="results/evo2_traitgym_v2",
        help="Directory containing {model}_{split}.parquet files.",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--output-metrics", default="results/evo2_traitgym_v2/metrics.parquet")
    p.add_argument("--output-markdown", default="results/evo2_traitgym_v2/results_table.md")
    args = p.parse_args()

    pred_dir = Path(args.predictions_dir)
    # Auto-discover all parquets matching the split. Avoids silently dropping
    # newly-supported models that aren't in the canonical MODELS list.
    parquets = sorted(pred_dir.glob(f"*_{args.split}.parquet"))
    assert parquets, (
        f"no predictions parquets found in {pred_dir} matching *_{args.split}.parquet"
    )

    discovered: list[str] = []
    all_metrics: list[pd.DataFrame] = []
    for path in parquets:
        model = path.stem.removesuffix(f"_{args.split}")
        discovered.append(model)
        print(f"[metrics] reading {path}  (model={model})")
        all_metrics.append(_metrics_for_model(path, model))

    metrics = pd.concat(all_metrics, ignore_index=True)

    # Display order: canonical models first (preserving MODELS order), then
    # any extras sorted alphabetically.
    known = [m for m in MODELS if m in discovered]
    extras = sorted(m for m in discovered if m not in MODELS)
    if extras:
        print(f"[metrics] models not in canonical MODELS list (appended): {extras}")
    model_order = known + extras

    out_metrics = Path(args.output_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_metrics, index=False)
    print(f"[metrics] wrote {out_metrics} ({len(metrics)} rows)")

    md = _build_markdown(metrics, model_order)
    out_md = Path(args.output_markdown)
    out_md.write_text(md)
    print(f"[metrics] wrote {out_md}")
    print("---")
    print(md)


if __name__ == "__main__":
    main()

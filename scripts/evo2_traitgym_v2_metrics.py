"""Aggregate Evo2 TraitGym v2 predictions into a single metrics table + markdown.

Reads predictions parquets written by scripts/evo2_traitgym_v2.py, reuses
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


MODELS = ["evo2_1b_base", "evo2_7b", "evo2_40b"]
SCORE_TYPES = ["minus_llr", "abs_llr"]


def _metrics_for_model(predictions_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(predictions_path)
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


def _build_markdown(metrics: pd.DataFrame) -> str:
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
    # Keep only the models we actually have, in canonical order
    cols = [m for m in MODELS if m in pivot.columns]
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
    all_metrics: list[pd.DataFrame] = []
    for model in MODELS:
        path = pred_dir / f"{model}_{args.split}.parquet"
        if not path.exists():
            print(f"[metrics] skipping {model}: {path} not found")
            continue
        print(f"[metrics] reading {path}")
        all_metrics.append(_metrics_for_model(path, model))

    assert all_metrics, "no predictions parquets found — nothing to aggregate"
    metrics = pd.concat(all_metrics, ignore_index=True)

    out_metrics = Path(args.output_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_metrics, index=False)
    print(f"[metrics] wrote {out_metrics} ({len(metrics)} rows)")

    md = _build_markdown(metrics)
    out_md = Path(args.output_markdown)
    out_md.write_text(md)
    print(f"[metrics] wrote {out_md}")
    print("---")
    print(md)


if __name__ == "__main__":
    main()

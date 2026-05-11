"""Generate iteration-1 analysis artifacts for the zeroshot_vep tracking issue.

Inputs:
- ``metrics_aggregated.parquet`` from the snakemake pipeline (with p_value + q_value).
- ``scratch/evals_v2_ref/_all.parquet`` — pre-fetched evals_v2 reference metrics
  for the sanity check.

Outputs (printed):
- Sanity check pass/fail vs. evals_v2 at native window.
- Top-N table per dataset (global_pooled, q < 0.05).
- Significance footprint (count of q < 0.05 per (dataset, aggregation)).
- Per-dataset heatmap PNGs in ``scratch/zeroshot_plots/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("mendelian_traits", "complex_traits", "eqtl")
LEADERBOARD_SCORE = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
    "eqtl": "abs_llr",
}
NATIVE_WINDOW = {
    "exp55-mammals": 256,
    "exp58-mammals": 256,
    "exp58-animals": 256,
    "exp59-mammals": 256,
    "exp136-proj_v30": 255,
}


def sanity_check(agg: pd.DataFrame, ref: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Compare per-subset values at native window vs. evals_v2 reference."""
    rows = []
    for model, w in NATIVE_WINDOW.items():
        for dataset, score in LEADERBOARD_SCORE.items():
            new = agg[
                (agg["model"] == model)
                & (agg["dataset"] == dataset)
                & (agg["window"] == w)
                & (agg["aggregation"] == "per_subset")
                & (agg["score"] == score)
            ][["subset", "value", "n_pairs"]].rename(
                columns={"value": "value_new", "n_pairs": "n_new"}
            )
            ref_sub = ref[
                (ref["model"] == model)
                & (ref["dataset"] == dataset)
                & (ref["score_type"] == score)
            ][["subset", "value", "n_pairs"]].rename(
                columns={"value": "value_ref", "n_pairs": "n_ref"}
            )
            m = new.merge(ref_sub, on="subset", how="outer")
            m["model"] = model
            m["dataset"] = dataset
            m["score"] = score
            m["abs_diff"] = (m["value_new"] - m["value_ref"]).abs()
            rows.append(m)
    cmp = pd.concat(rows, ignore_index=True)
    return float(cmp["abs_diff"].max()), cmp


def top_n_per_dataset(agg: pd.DataFrame, n: int = 10, q_cutoff: float = 0.05) -> dict[str, pd.DataFrame]:
    """Top N (model, window, score) tuples per dataset on global_pooled, sig-filtered."""
    out: dict[str, pd.DataFrame] = {}
    pool = agg[agg["aggregation"] == "global_pooled"]
    for ds in DATASETS:
        ds_pool = pool[(pool["dataset"] == ds) & (pool["q_value"] < q_cutoff)]
        top = ds_pool.sort_values("value", ascending=False).head(n)
        out[ds] = top[
            ["model", "window", "score", "value", "se", "n_pairs", "p_value", "q_value"]
        ].copy()
    return out


def best_score_per_subset(agg: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, subset), find the best (score, model, window) by value.

    Only includes rows where q_value < 0.05 (else no signal).
    """
    per = agg[(agg["aggregation"] == "per_subset") & (agg["q_value"] < 0.05)].copy()
    if len(per) == 0:
        return pd.DataFrame()
    idx = per.groupby(["dataset", "subset"])["value"].idxmax()
    best = per.loc[idx, ["dataset", "subset", "score", "model", "window", "value", "n_pairs", "q_value"]]
    return best.sort_values(["dataset", "subset"]).reset_index(drop=True)


def likelihood_vs_embedding_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Compare mean PairwiseAccuracy of likelihood vs. embedding scores per dataset.

    Aggregates over (model, window) for global_pooled rows; reports mean ± stdev
    of the per (model, window) PairwiseAccuracy for each score-family.
    """
    LIKELIHOOD = ("llr", "minus_llr", "abs_llr", "minus_logp_ref", "minus_logp_alt", "entropy")
    pool = agg[agg["aggregation"] == "global_pooled"].copy()
    pool["family"] = pool["score"].apply(
        lambda s: "likelihood" if s in LIKELIHOOD else "embedding"
    )
    rows = []
    for (ds, fam), sub in pool.groupby(["dataset", "family"]):
        rows.append({
            "dataset": ds,
            "family": fam,
            "n_score_method_combos": len(sub),
            "mean_value": float(sub["value"].mean()),
            "median_value": float(sub["value"].median()),
            "best_value": float(sub["value"].max()),
            "frac_q_lt_0.05": float((sub["q_value"] < 0.05).mean()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "family"])


def window_effect_per_dataset(agg: pd.DataFrame) -> pd.DataFrame:
    """How much does window size matter? Per dataset, compare best score for
    each window. Looks at global_pooled, taking the max value across (model, score)
    for each window.
    """
    pool = agg[agg["aggregation"] == "global_pooled"].copy()
    best_per = pool.groupby(["dataset", "window"])["value"].max().reset_index()
    pivot = best_per.pivot(index="dataset", columns="window", values="value")
    pivot.columns = [f"best_w{c}" for c in pivot.columns]
    return pivot.reset_index()


def significance_footprint(agg: pd.DataFrame, q_cutoff: float = 0.05) -> pd.DataFrame:
    """Count of q < cutoff per (dataset, aggregation), out of total tests."""
    rows = []
    for ds in DATASETS:
        for agg_kind in ("per_subset", "global_pooled", "global_macro"):
            sub = agg[(agg["dataset"] == ds) & (agg["aggregation"] == agg_kind)]
            sub_valid = sub.dropna(subset=["q_value"])
            n_sig = int((sub_valid["q_value"] < q_cutoff).sum())
            n_total = len(sub_valid)
            rows.append(
                {
                    "dataset": ds,
                    "aggregation": agg_kind,
                    "n_tests": n_total,
                    f"n_q_lt_{q_cutoff}": n_sig,
                    "frac_sig": n_sig / n_total if n_total > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def render_heatmaps(agg: pd.DataFrame, out_dir: Path) -> None:
    """Render PNG heatmaps per dataset. Skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available — skipping heatmaps", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    agg = agg.copy()
    agg["mw"] = agg["model"] + " | w" + agg["window"].astype(str)

    for ds in DATASETS:
        sub_ds = agg[agg["dataset"] == ds]
        for agg_kind in ("global_pooled", "global_macro"):
            df = sub_ds[sub_ds["aggregation"] == agg_kind]
            if len(df) == 0:
                continue
            pivot = df.pivot_table(
                index="score", columns="mw", values="value", aggfunc="first"
            )
            sig_pivot = df.pivot_table(
                index="score", columns="mw", values="q_value", aggfunc="first"
            )
            fig, ax = plt.subplots(
                figsize=(max(8, 0.55 * pivot.shape[1]), max(7, 0.32 * pivot.shape[0]))
            )
            im = ax.imshow(
                pivot.values, vmin=0.4, vmax=0.85, cmap="RdBu_r", aspect="auto"
            )
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index, fontsize=8)
            # Annotate cells: value, mark q<0.05 with bold *.
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    q = sig_pivot.values[i, j]
                    mark = "*" if (not np.isnan(q) and q < 0.05) else ""
                    ax.text(
                        j, i, f"{v:.2f}{mark}",
                        ha="center", va="center", fontsize=6,
                        color="black" if 0.45 < v < 0.62 else "white",
                    )
            plt.colorbar(im, ax=ax, label="PairwiseAccuracy")
            ax.set_title(f"{ds} — {agg_kind}  (* = q<0.05, BH-FDR within family)")
            plt.tight_layout()
            plt.savefig(out_dir / f"{ds}_{agg_kind}.png", dpi=140)
            plt.close(fig)
            print(f"[plots] wrote {out_dir / f'{ds}_{agg_kind}.png'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "agg_parquet",
        help="Path or s3:// URI to metrics_aggregated.parquet",
    )
    ap.add_argument(
        "--ref-parquet",
        default="scratch/evals_v2_ref/_all.parquet",
        help="evals_v2 reference parquet for sanity check",
    )
    ap.add_argument("--out-dir", default="scratch/zeroshot_plots")
    args = ap.parse_args(argv)

    agg = pd.read_parquet(args.agg_parquet)
    print(f"[load] {len(agg)} rows from {args.agg_parquet}")

    # Sanity check.
    ref_path = Path(args.ref_parquet)
    if ref_path.exists():
        max_diff, cmp_df = sanity_check(agg, pd.read_parquet(ref_path))
        print(f"\n[sanity] max abs diff vs. evals_v2 (native window, leaderboard scores): {max_diff:.6f}")
        bad = cmp_df[cmp_df["abs_diff"] > 0.01]
        if len(bad):
            print("[sanity] FAIL — diffs > 0.01:")
            print(bad.to_string(index=False))
        else:
            print(f"[sanity] PASS (all {len(cmp_df)} per-subset comparisons within 0.01)")
    else:
        print(f"[sanity] no reference at {ref_path}; skipping")

    # Top-N per dataset.
    print("\n=== TOP-10 per dataset (global_pooled, q < 0.05) ===")
    for ds, top in top_n_per_dataset(agg).items():
        print(f"\n--- {ds} ---")
        if len(top) == 0:
            print("  no rows pass q < 0.05 — dataset/scores under-powered or null")
        else:
            print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Significance footprint.
    print("\n=== Significance footprint (q < 0.05 per (dataset, aggregation)) ===")
    print(significance_footprint(agg).to_string(index=False))

    # Best (score, model, window) per (dataset, subset).
    print("\n=== Best (score, model, window) per (dataset, subset), q < 0.05 only ===")
    best = best_score_per_subset(agg)
    if len(best) > 0:
        print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("  no significant rows")

    # Likelihood vs embedding family summary.
    print("\n=== Likelihood vs embedding score family (global_pooled) ===")
    fam = likelihood_vs_embedding_summary(agg)
    print(fam.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Window effect.
    print("\n=== Best PairwiseAccuracy per (dataset, window), global_pooled ===")
    print(window_effect_per_dataset(agg).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Heatmaps.
    print(f"\n=== Rendering heatmaps to {args.out_dir} ===")
    render_heatmaps(agg, Path(args.out_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())

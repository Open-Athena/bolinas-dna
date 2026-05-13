"""Fast iter-1b experiments to probe the high-D dilution hypothesis from iter-1.

Iter-1 finding: BFS supervised classifiers underperform the zero-shot
``embed_last_l2`` baseline on 2 of 3 datasets. This script tests two
orthogonal angles:

1. **Alternative zero-shot scalars** computed from the cached pooled features:
   * pooled_l2 = ||mean_ref − mean_alt||
   * pooled_cosine_dist = 1 − cos(mean_ref, mean_alt)
   * pooled_dot = <mean_ref, mean_alt> (asymmetric in sign, so use −|dot|)
   * sum_innerprod = sum over channels of traitgym_innerprod
   If any of these beats ``embed_last_l2`` macro PA, the zero-shot ceiling
   was under-stated.

2. **Single-feature supervised baselines**: chrom-grouped OOF LogReg on a
   single scalar (e.g. ``embed_last_l2`` alone). If the single-feature OOF
   ≈ zero-shot, it proves the BFS supervised head's high-D features are
   dilution, not extra signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolinas.evals.metrics import compute_pairwise_metrics
from bolinas.supervised.cv import oof_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ["mendelian_traits", "complex_traits", "eqtl"]


def load_features(ds: str) -> pd.DataFrame:
    path = (
        f"s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/"
        f"exp166-p1B/{ds}.parquet"
    )
    return pd.read_parquet(path)


def pooled_zeroshot_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extra zero-shot scalar features from the cached pools."""
    mean_ref = np.stack(df["mean_ref"].to_numpy()).astype(np.float64)
    mean_alt = np.stack(df["mean_alt"].to_numpy()).astype(np.float64)
    innerprod = np.stack(df["traitgym_innerprod"].to_numpy()).astype(np.float64)

    pooled_l2 = np.linalg.norm(mean_alt - mean_ref, axis=1)
    cos_num = (mean_ref * mean_alt).sum(axis=1)
    cos_den = (
        np.linalg.norm(mean_ref, axis=1) * np.linalg.norm(mean_alt, axis=1) + 1e-12
    )
    pooled_cosine_dist = 1.0 - cos_num / cos_den
    pooled_dot_neg_abs = -np.abs((mean_ref * mean_alt).sum(axis=1))
    sum_innerprod_abs = np.abs(innerprod.sum(axis=1))

    return pd.DataFrame(
        {
            "pooled_l2": pooled_l2.astype(np.float32),
            "pooled_cosine_dist": pooled_cosine_dist.astype(np.float32),
            "pooled_dot_neg_abs": pooled_dot_neg_abs.astype(np.float32),
            "sum_innerprod_abs": sum_innerprod_abs.astype(np.float32),
        }
    )


def macro_pa(df: pd.DataFrame, scores: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a tidy macro_avg + _global_ slice."""
    meta = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    m = compute_pairwise_metrics(
        dataset=df[[c for c in meta if c in df.columns]],
        scores=scores[cols],
        score_columns=cols,
    )
    return m[m["subset"].isin(["_macro_avg_", "_global_"])][
        ["score_type", "subset", "value", "se", "n_pairs"]
    ]


def single_feature_oof(
    df: pd.DataFrame, score_col: str
) -> tuple[np.ndarray, list[dict]]:
    """Chrom-grouped 3-fold OOF on a single-scalar feature.

    Sigmoid LogReg with class_weight balanced; tiny C grid since a single
    feature has no real risk of over-fitting.
    """
    X = df[[score_col]].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                ),
            ),
        ]
    )
    return oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=estimator,
        param_grid={"clf__C": [1.0]},  # any C works for a single feature
        n_splits=3,
        n_splits_inner=3,
        scoring="average_precision",
    )


def main():
    all_rows = []
    for ds in DATASETS:
        print(f"=== {ds} ===")
        feats = load_features(ds)
        extras = pooled_zeroshot_scalars(feats)
        wide = pd.concat([feats.reset_index(drop=True), extras], axis=1)

        zero_cols = [
            "minus_llr",
            "abs_llr",
            "embed_last_l2",
            "pooled_l2",
            "pooled_cosine_dist",
            "pooled_dot_neg_abs",
            "sum_innerprod_abs",
        ]
        m_zero = macro_pa(wide, wide, zero_cols)
        m_zero["dataset"] = ds
        m_zero["family"] = "baseline_extra"
        print(m_zero.to_string(index=False))
        all_rows.append(m_zero)

        print(f"\n-- single-feature OOF on {ds} --")
        for col in ["embed_last_l2", "pooled_l2", "pooled_cosine_dist"]:
            preds, _ = single_feature_oof(wide, col)
            preds_df = wide[
                ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
            ].copy()
            preds_df[col + "_oof"] = preds
            m_oof = macro_pa(
                preds_df, preds_df[[col + "_oof"]], [col + "_oof"]
            )
            m_oof["dataset"] = ds
            m_oof["family"] = "single_feature_oof"
            print(m_oof.to_string(index=False))
            all_rows.append(m_oof)

        print()

    out = pd.concat(all_rows, ignore_index=True)
    out.to_parquet("scratch/iter1b_fast_baselines.parquet", index=False)
    print(f"\nSaved scratch/iter1b_fast_baselines.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()

"""Two-scalar supervised OOF combinations to test additive signal.

If ``embed_last_l2`` is already the optimal compression, combining it with
``minus_llr``/``abs_llr`` via a 2-D LogReg shouldn't help — they're capturing
the same variant-effect signal. If it *does* help materially, the zero-shot
recipe-composition story from #175 (e.g. ``rank-mean(minus_llr,
embed_l2_flat_last, alphagenome_max_l2)``) generalises to a single linear
combination.
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
    return pd.read_parquet(
        f"s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/"
        f"exp166-p1B/{ds}.parquet"
    )


def k_feature_oof(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=np.float64)
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
    preds, _ = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=estimator,
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
        n_splits=3,
        n_splits_inner=3,
    )
    return preds


def macro_global(
    df: pd.DataFrame, scores: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    meta = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    m = compute_pairwise_metrics(
        dataset=df[[c for c in meta if c in df.columns]],
        scores=scores[cols],
        score_columns=cols,
    )
    return m[m["subset"].isin(["_macro_avg_", "_global_"])][
        ["score_type", "subset", "value", "se"]
    ]


def main():
    rows = []
    for ds in DATASETS:
        df = load_features(ds)
        # Build the combined-scalar OOF predictions for a few small recipes.
        combos = [
            ["embed_last_l2", "abs_llr"],
            ["embed_last_l2", "minus_llr"],
            ["embed_last_l2", "abs_llr", "minus_llr"],
        ]
        scored = df[
            ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
        ].copy()
        for cols in combos:
            name = "+".join(cols)
            scored[name] = k_feature_oof(df, cols)
        m = macro_global(scored, scored, ["+".join(c) for c in combos])
        m["dataset"] = ds
        print(f"=== {ds} ===")
        print(m.to_string(index=False))
        print()
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet("scratch/iter1b_scalar_combos.parquet", index=False)


if __name__ == "__main__":
    main()

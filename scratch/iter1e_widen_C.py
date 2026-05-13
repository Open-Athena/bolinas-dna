"""Iter-1e: extend the C grid downward — every selected C in iter-1d hit the
boundary at 0.0001. Check whether the optimum sits even lower.

Same datasets, same 2-scalar / 3-scalar / sym_concat probes, but now with
C in logspace(-10, 0, 11) for both standard logreg and pair-aware.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolinas.evals.metrics import compute_pairwise_metrics
from bolinas.supervised.classifiers import pairwise_oof_predict
from bolinas.supervised.cv import oof_predict
from bolinas.supervised.features import build_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
WIDE_C = tuple(10.0 ** np.arange(-10, 1))  # 11 values


def load(ds):
    return pd.read_parquet(
        f"s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/"
        f"exp166-p1B/{ds}.parquet"
    )


def macro_pa(df, scores, name):
    df = df.copy()
    df[name] = scores
    m = compute_pairwise_metrics(
        dataset=df[META], scores=df[[name]], score_columns=[name]
    )
    return (
        float(m[m["subset"] == "_macro_avg_"]["value"].iloc[0]),
        float(m[m["subset"] == "_macro_avg_"]["se"].iloc[0]),
        float(m[m["subset"] == "_global_"]["value"].iloc[0]),
    )


def standard_logreg_oof(df, cols, c_grid):
    X = df[cols].to_numpy(dtype=np.float64) if isinstance(cols, list) else cols
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
        param_grid={"clf__C": list(c_grid)},
        n_splits=3,
        n_splits_inner=3,
    )


def pa_oof(df, X, c_grid):
    y = df["label"].astype(int).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    mg = df["match_group"].to_numpy()
    return pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=tuple(c_grid),
        n_splits=3,
        n_splits_inner=3,
    )


def selected(records):
    return [r["best_params"].get("C", r["best_params"].get("clf__C")) for r in records]


def main():
    rows = []
    for ds in ["mendelian_traits", "complex_traits", "eqtl"]:
        print(f"=== {ds} ===")
        df = load(ds)
        for cols in [
            ["embed_last_l2", "minus_llr"],
            ["embed_last_l2", "minus_llr", "abs_llr"],
        ]:
            # Standard logreg.
            preds, recs = standard_logreg_oof(df, cols, WIDE_C)
            mp = macro_pa(df, preds, "_")
            cs = selected(recs)
            print(
                f"  std    {'+'.join(cols):60s} macro={mp[0]:.4f} ± {mp[1]:.4f}  global={mp[2]:.4f}  Cs={cs}"
            )
            rows.append(
                dict(dataset=ds, recipe="+".join(cols), classifier="logreg_l2_xwide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs)
            )
            # Pair-aware.
            preds, recs = pa_oof(df, df[cols].to_numpy(dtype=np.float64), WIDE_C)
            mp = macro_pa(df, preds, "_")
            cs = selected(recs)
            print(
                f"  pa     {'+'.join(cols):60s} macro={mp[0]:.4f} ± {mp[1]:.4f}  global={mp[2]:.4f}  Cs={cs}"
            )
            rows.append(
                dict(dataset=ds, recipe="+".join(cols), classifier="pairwise_logreg_xwide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs)
            )

    # High-D mendelian × sym_concat.
    print("\n=== mendelian × sym_concat (D=3840) ===")
    df = load("mendelian_traits")
    X = build_features(df, "sym_concat")
    preds, recs = pa_oof(df, X, WIDE_C)
    mp = macro_pa(df, preds, "_")
    cs = selected(recs)
    print(f"  pa-xwide sym_concat macro={mp[0]:.4f} ± {mp[1]:.4f}  global={mp[2]:.4f}  Cs={cs}")
    rows.append(
        dict(dataset="mendelian_traits", recipe="sym_concat", classifier="pairwise_logreg_xwide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs)
    )

    pd.DataFrame(rows).to_parquet("scratch/iter1e_xwide.parquet", index=False)
    print("\nSaved scratch/iter1e_xwide.parquet")


if __name__ == "__main__":
    main()

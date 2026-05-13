"""Iter-1d: deeper look at pair-aware logreg vs rank-mean / standard LogReg.

Pair-aware (RankNet with logistic link) directly optimises the metric we
care about — within-``match_group`` ranking — by fitting on pair-difference
vectors. Iter-1's BFS used a single ``C=1.0`` for pair-aware; let's widen
the grid and probe a few targeted feature sets:

1. **2-scalar mendelian**: ``embed_last_l2 + minus_llr`` — rank-mean hit
   0.7682 on mendelian here; does pair-aware logreg with a tuned C match
   or beat?
2. **3-scalar across all datasets**: ``embed_last_l2 + minus_llr + abs_llr``
   — standard LogReg-OOF got 0.7589 on mendelian; can pair-aware do better?
3. **High-D mendelian × sym_concat**: iter-1's BFS pair-aware got 0.474
   here. With a wider C grid on the same features (D=3840), does the
   ranking loss still tank in the high-D regime, or does C-tuning save it?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from bolinas.evals.metrics import compute_pairwise_metrics
from bolinas.supervised.classifiers import pairwise_oof_predict
from bolinas.supervised.cv import oof_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ["mendelian_traits", "complex_traits", "eqtl"]
META = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
WIDE_C = tuple(10.0 ** np.arange(-4, 4))  # logspace(-4, 3, 8)


def load(ds):
    return pd.read_parquet(
        f"s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/"
        f"exp166-p1B/{ds}.parquet"
    )


def add_scalars(df):
    df = df.copy()
    mr = np.stack(df["mean_ref"].to_numpy())
    ma = np.stack(df["mean_alt"].to_numpy())
    df["pooled_l2"] = np.linalg.norm(ma - mr, axis=1).astype(np.float32)
    cn = (mr * ma).sum(axis=1)
    cd = np.linalg.norm(mr, axis=1) * np.linalg.norm(ma, axis=1) + 1e-12
    df["pooled_cosine_dist"] = (1.0 - cn / cd).astype(np.float32)
    return df


def macro_pa(df, scores, name):
    df = df.copy()
    df[name] = scores
    m = compute_pairwise_metrics(
        dataset=df[META], scores=df[[name]], score_columns=[name]
    )
    macro = m[m["subset"] == "_macro_avg_"].iloc[0]
    glob = m[m["subset"] == "_global_"].iloc[0]
    return float(macro["value"]), float(macro["se"]), float(glob["value"])


def rank_mean(df, cols):
    return np.column_stack([rankdata(df[c]) for c in cols]).mean(axis=1)


def standard_logreg_oof(df, cols, c_grid):
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
    preds, records = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=estimator,
        param_grid={"clf__C": list(c_grid)},
        n_splits=3,
        n_splits_inner=3,
    )
    return preds, records


def pairwise_oof_with_grid(df, cols, c_grid):
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    mg = df["match_group"].to_numpy()
    preds, records = pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=tuple(c_grid),
        n_splits=3,
        n_splits_inner=3,
    )
    return preds, records


def selected_C_summary(records):
    """One Cs-per-fold value, useful to know if the grid boundary was hit."""
    return [r["best_params"].get("C", r["best_params"].get("clf__C")) for r in records]


def main():
    rows = []

    # Experiments 1 & 2: scalars only.
    for ds in DATASETS:
        df = add_scalars(load(ds))
        # 2-scalar combo (the mendelian winner from iter-1b).
        cols2 = ["embed_last_l2", "minus_llr"]
        # 3-scalar.
        cols3 = ["embed_last_l2", "minus_llr", "abs_llr"]

        print(f"=== {ds} ===")

        # Rank-mean (reference).
        for cols in [cols2, cols3]:
            rm = rank_mean(df, cols)
            mp = macro_pa(df, rm, f"rm:{'+'.join(cols)}")
            print(f"  rank-mean        {'+'.join(cols):60s} macro={mp[0]:.4f} ± {mp[1]:.4f}")
            rows.append(dict(dataset=ds, recipe="+".join(cols), classifier="rank_mean", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=None))

        # Standard LogReg (wide C) — baseline supervised.
        for cols in [cols2, cols3]:
            preds, recs = standard_logreg_oof(df, cols, WIDE_C)
            mp = macro_pa(df, preds, f"std:{'+'.join(cols)}")
            cs = selected_C_summary(recs)
            print(f"  std logreg       {'+'.join(cols):60s} macro={mp[0]:.4f} ± {mp[1]:.4f}  Cs={cs}")
            rows.append(dict(dataset=ds, recipe="+".join(cols), classifier="logreg_l2_wide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs))

        # Pair-aware LogReg (wide C).
        for cols in [cols2, cols3]:
            preds, recs = pairwise_oof_with_grid(df, cols, WIDE_C)
            mp = macro_pa(df, preds, f"pa:{'+'.join(cols)}")
            cs = selected_C_summary(recs)
            print(f"  pair-aware       {'+'.join(cols):60s} macro={mp[0]:.4f} ± {mp[1]:.4f}  Cs={cs}")
            rows.append(dict(dataset=ds, recipe="+".join(cols), classifier="pairwise_logreg_wide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs))

        print()

    # Experiment 3: high-D mendelian × sym_concat with wider pair-aware C.
    print("=== mendelian × sym_concat (D=3840) — high-D pair-aware deep-dive ===")
    df = load("mendelian_traits")
    from bolinas.supervised.features import build_features
    X = build_features(df, "sym_concat")
    y = df["label"].astype(int).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    mg = df["match_group"].to_numpy()

    preds, recs = pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=tuple(WIDE_C),
        n_splits=3,
        n_splits_inner=3,
    )
    mp = macro_pa(df, preds, "pa-wide_sym_concat")
    cs = selected_C_summary(recs)
    print(f"  pair-aware sym_concat (wide C) macro={mp[0]:.4f} ± {mp[1]:.4f}  Cs={cs}")
    rows.append(dict(dataset="mendelian_traits", recipe="sym_concat", classifier="pairwise_logreg_wide", macro=mp[0], macro_se=mp[1], **{"global": mp[2]}, selected_Cs=cs))

    pd.DataFrame(rows).to_parquet("scratch/iter1d_pairaware.parquet", index=False)
    print("\nSaved scratch/iter1d_pairaware.parquet")


if __name__ == "__main__":
    main()

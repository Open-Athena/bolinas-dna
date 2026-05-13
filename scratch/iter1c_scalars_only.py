"""Iter-1c: every iter-1 classifier on scalars only (no high-D pool blocks).

Tests: when we strip out the D=1920+ pool features and feed the classifiers
just the 5-6 zero-shot scalars, can any of them match the rank-mean recipe
from iter-1b?

Features used (5 scalars, no high-D blocks):
* embed_last_l2 (flattened-sequence L2)
* minus_llr (signed)
* abs_llr
* pooled_l2 (mean-pool L2)
* pooled_cosine_dist (mean-pool cosine)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bolinas.evals.metrics import compute_pairwise_metrics
from bolinas.supervised.classifiers import all_standard_specs, pairwise_oof_predict
from bolinas.supervised.cv import oof_predict


DATASETS = ["mendelian_traits", "complex_traits", "eqtl"]
SCALAR_COLS = [
    "embed_last_l2",
    "minus_llr",
    "abs_llr",
    "pooled_l2",
    "pooled_cosine_dist",
]
META = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]


def load_features(ds: str) -> pd.DataFrame:
    return pd.read_parquet(
        f"s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/"
        f"exp166-p1B/{ds}.parquet"
    )


def add_pooled_scalars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mr = np.stack(df["mean_ref"].to_numpy())
    ma = np.stack(df["mean_alt"].to_numpy())
    df["pooled_l2"] = np.linalg.norm(ma - mr, axis=1).astype(np.float32)
    cn = (mr * ma).sum(axis=1)
    cd = np.linalg.norm(mr, axis=1) * np.linalg.norm(ma, axis=1) + 1e-12
    df["pooled_cosine_dist"] = (1.0 - cn / cd).astype(np.float32)
    return df


def macro_global(df: pd.DataFrame, scores: np.ndarray, name: str) -> dict:
    df = df.copy()
    df[name] = scores
    m = compute_pairwise_metrics(
        dataset=df[META], scores=df[[name]], score_columns=[name]
    )
    return {
        "macro_avg": m[m["subset"] == "_macro_avg_"]["value"].iloc[0],
        "macro_se": m[m["subset"] == "_macro_avg_"]["se"].iloc[0],
        "global": m[m["subset"] == "_global_"]["value"].iloc[0],
        "global_se": m[m["subset"] == "_global_"]["se"].iloc[0],
    }


def main():
    rows = []
    for ds in DATASETS:
        print(f"=== {ds} ===")
        df = add_pooled_scalars(load_features(ds))
        X = df[SCALAR_COLS].to_numpy(dtype=np.float64)
        y = df["label"].astype(int).to_numpy()
        chroms = df["chrom"].astype(str).to_numpy()
        mg = df["match_group"].to_numpy()

        # Each standard classifier.
        for spec in all_standard_specs(mode="bfs"):
            preds, _ = oof_predict(
                X=X,
                y=y,
                chroms=chroms,
                estimator=spec.estimator,
                param_grid=spec.param_grid,
                n_splits=3,
                n_splits_inner=3,
            )
            row = macro_global(df, preds, spec.name)
            row.update(dataset=ds, classifier=spec.name, family="scalars_only_oof")
            rows.append(row)
            print(f"  {spec.name:18s} macro={row['macro_avg']:.4f} ± {row['macro_se']:.4f}  global={row['global']:.4f}")

        # Pair-aware variant.
        preds, _ = pairwise_oof_predict(
            X=X,
            y=y,
            chroms=chroms,
            match_group=mg,
            base="logreg",
            C_grid=(1.0,),
            n_splits=3,
            n_splits_inner=3,
        )
        row = macro_global(df, preds, "pairwise_logreg")
        row.update(dataset=ds, classifier="pairwise_logreg", family="scalars_only_oof")
        rows.append(row)
        print(f"  {'pairwise_logreg':18s} macro={row['macro_avg']:.4f} ± {row['macro_se']:.4f}  global={row['global']:.4f}")
        print()

    out = pd.DataFrame(rows)
    out.to_parquet("scratch/iter1c_scalars_only.parquet", index=False)
    print(f"\nSaved scratch/iter1c_scalars_only.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()

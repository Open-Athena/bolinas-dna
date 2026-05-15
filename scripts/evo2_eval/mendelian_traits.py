"""Score Mendelian-traits matched-pair variants with an Evo2 model.

Entry for issue #131's leaderboard ride-along (issue #161). Per-variant
score bundle (LLR + next-token JSD) over an 8192-bp window, FWD+RC
averaged by default, followed by PairwiseAccuracy ± SE per consequence
subset. One model per run.

Single-GPU execution only — Evo2's Vortex backend handles its own
multi-GPU sharding (e.g. 40B on GH200's 96 GB sits on one device). We do
not use ``torchrun`` here, matching the cluster yaml's single-process
convention.
"""

# --- header guard: must run before importing torch / evo2 / biofoundation ---
# From biofoundation/examples/evo2_llr.py. Kept for parity with the LL-gap
# script in case this entry is ever invoked under torchrun; under plain
# ``python`` it's a no-op (LOCAL_RANK unset).
import os

_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank))
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

from bolinas.pipelines.evals.evo2 import EVO2_MODEL_CHOICES  # noqa: E402
from bolinas.pipelines.evals.metrics import (  # noqa: E402
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_pairwise_metrics,
)

# Script-local Evo2 scoring (no KV-cache; doesn't go through
# bolinas.model.runner / HF Trainer — see scripts/evo2_eval/_evo2_scoring.py
# docstring for why).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evo2_scoring import compute_evo2_bundle  # noqa: E402, I001


DATASET_HF_PATH = "bolinas-dna/evals_mendelian_traits"
SCORE_COLUMN = "minus_llr"
DATASET_NAME = "mendelian_traits"
# Schema of the matched-pair eval datasets. Same tuple as
# `bolinas.pipelines.evals.conservation.REQUIRED_VARIANT_COLUMNS`, hardcoded
# here to avoid importing conservation.py (which triggers a top-level
# `import pyBigWig` — needs libBigWig.so on the host).
REQUIRED_VARIANT_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=EVO2_MODEL_CHOICES)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--genome-path",
        default="results/genome.fa.gz",
        help="Path to GRCh38 reference (Ensembl release-113 primary assembly).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Scores parquet path. "
        "Defaults to results/evo2_mendelian_traits/{model}_{split}.parquet",
    )
    p.add_argument(
        "--output-metrics",
        default=None,
        help="Metrics parquet path. "
        "Defaults to results/evo2_mendelian_traits/{model}_{split}_metrics.parquet",
    )
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-batch row count. Each batch feeds the model "
        "[2*batch_size, window_size] tokens (ref+alt concatenated). "
        "Default 16 worked for evo2_1b_base on GH200 at window=8192; "
        "halve for 7B, more for 40B-needs-bs=1.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N variants. Smoke-test flag.",
    )
    p.add_argument(
        "--no-rc-avg",
        action="store_true",
        help="Disable FWD+RC averaging (ablation only). Default is RC on, "
        "matching evals_v2 and the #161 leaderboard protocol.",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/evo2_mendelian_traits/{args.model}_{args.split}.parquet"
    if args.output_metrics is None:
        args.output_metrics = (
            f"results/evo2_mendelian_traits/{args.model}_{args.split}_metrics.parquet"
        )

    ds = load_dataset(DATASET_HF_PATH, split=args.split).to_pandas()
    missing = [c for c in REQUIRED_VARIANT_COLUMNS if c not in ds.columns]
    assert not missing, f"dataset is missing required columns: {missing}"
    assert ds["label"].isna().sum() == 0, "label column contains NaN"
    assert set(ds["label"].astype(int).unique()) <= {0, 1}, "label is not binary 0/1"
    if args.limit is not None:
        ds = ds.head(args.limit).reset_index(drop=True)
        print(f"[evo2] --limit {args.limit} applied, scoring {len(ds)} variants")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        rc_state = "OFF" if args.no_rc_avg else "ON"
        print(
            f"[evo2] model={args.model} split={args.split} "
            f"n={len(ds)} rc_avg={rc_state} window={args.window_size}"
        )
        print(f"[evo2] subsets:\n{ds['subset'].value_counts().to_string()}")
        n_pairs_input = ds["match_group"].nunique()
        print(f"[evo2] match_groups (pairs): {n_pairs_input}")

    scores = compute_evo2_bundle(
        model_name=args.model,
        df=ds[["chrom", "pos", "ref", "alt"]],
        genome_path=args.genome_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        rc_avg=not args.no_rc_avg,
    )

    if rank != 0:
        return

    out = pd.concat(
        [
            ds[list(REQUIRED_VARIANT_COLUMNS)].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    assert len(out) == len(ds)
    assert np.isfinite(out["llr"]).all()
    assert np.isfinite(out["next_token_jsd_mean"]).all()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        f"[evo2] wrote {out_path}  "
        f"llr min={out.llr.min():.3f} max={out.llr.max():.3f} "
        f"mean={out.llr.mean():.3f}  "
        f"jsd mean={out.next_token_jsd_mean.mean():.4f}"
    )

    metrics = compute_pairwise_metrics(
        dataset=out[list(REQUIRED_VARIANT_COLUMNS)],
        scores=out[[SCORE_COLUMN]],
        score_columns=[SCORE_COLUMN],
    )
    metrics["model"] = args.model
    metrics["dataset"] = DATASET_NAME
    metrics["split"] = args.split

    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    print(f"[evo2] wrote {metrics_path}")

    glb = metrics[metrics["subset"] == GLOBAL_SUBSET].iloc[0]
    mac = metrics[metrics["subset"] == MACRO_AVG_SUBSET].iloc[0]
    print(
        f"[evo2] {args.model} PairwiseAccuracy on '{SCORE_COLUMN}':\n"
        f"  Global     PA = {glb['value']:.3f} ± {glb['se']:.3f}  "
        f"(n_pairs={int(glb['n_pairs'])})\n"
        f"  Macro Avg  PA = {mac['value']:.3f} ± {mac['se']:.3f}  "
        f"(K={int(mac['n_pairs'])} subsets ≥ n_min)"
    )
    per_subset = metrics[
        ~metrics["subset"].isin([GLOBAL_SUBSET, MACRO_AVG_SUBSET])
    ].sort_values("n_pairs", ascending=False)
    for _, row in per_subset.iterrows():
        print(
            f"  {row['subset']:36s} PA = {row['value']:.3f} ± {row['se']:.3f}  "
            f"(n_pairs={int(row['n_pairs'])}, n_ties={int(row['n_ties'])})"
        )


if __name__ == "__main__":
    main()

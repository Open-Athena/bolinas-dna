"""Score TraitGym Mendelian v2 variants with an Evo2 model.

Entry for issue #131. Per-variant LLR only, 8192-bp context, one model per run.

Always launch this with plain ``python``. Do NOT use ``torchrun`` / HF DDP —
Evo2's Vortex backend handles its own multi-GPU sharding internally, and
layering HF's data parallelism on top fights Vortex's device placement. For
small models (1B, 7B) on a multi-GPU node, the extra GPUs sit idle. That's
intended.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from bolinas.evals.evo2 import compute_evo2_llr, scores_dataframe


DATASET_HF_PATH = "bolinas-dna/evals-traitgym_mendelian_v2"
REQUIRED_COLS = ["chrom", "pos", "ref", "alt", "label", "subset", "consequence_group"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        choices=["evo2_1b_base", "evo2_7b", "evo2_40b"],
    )
    p.add_argument("--split", default="train")
    p.add_argument(
        "--genome-path",
        default="results/genome.fa.gz",
        help="Path to GRCh38 reference (Ensembl release-113 primary assembly).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Defaults to results/evo2_traitgym_v2/{model}_{split}.parquet",
    )
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/evo2_traitgym_v2/{args.model}_{args.split}.parquet"

    ds = load_dataset(DATASET_HF_PATH, split=args.split).to_pandas()
    missing = [c for c in REQUIRED_COLS if c not in ds.columns]
    assert not missing, f"dataset is missing required columns: {missing}"
    assert ds["label"].isna().sum() == 0, "label column contains NaN"
    assert set(ds["label"].unique()) <= {0, 1}, "label is not binary 0/1"

    print(f"[evo2] model={args.model} split={args.split} n={len(ds)}")
    print(f"[evo2] subsets:\n{ds['subset'].value_counts().to_string()}")

    llr = compute_evo2_llr(
        model_name=args.model,
        dataset=ds[["chrom", "pos", "ref", "alt"]],
        genome_path=args.genome_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    scores = scores_dataframe(llr)
    out = pd.concat(
        [
            ds[REQUIRED_COLS].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    assert len(out) == len(ds)
    assert np.isfinite(out["llr"]).all()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(
        f"[evo2] wrote {out_path}  "
        f"llr min={out.llr.min():.3f} max={out.llr.max():.3f} mean={out.llr.mean():.3f}"
    )


if __name__ == "__main__":
    main()

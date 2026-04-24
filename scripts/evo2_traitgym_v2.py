"""Score TraitGym Mendelian v2 variants with an Evo2 model.

Entry for issue #131. Per-variant LLR only, 8192-bp context, one model per run.

Two launch modes:

1. Plain ``python`` — single-GPU inference. Evo2's Vortex backend uses only
   whatever's visible via ``CUDA_VISIBLE_DEVICES``. For small models (1B, 7B)
   this runs on one GPU; other GPUs idle.

2. ``torchrun --nproc_per_node=N`` — data-parallel across N GPUs. Each rank
   pins to one GPU via the ``LOCAL_RANK`` header below (copied verbatim from
   biofoundation/examples/evo2_llr.py), so Evo2/Vortex inside each rank only
   sees that one GPU and HF Trainer's DDP wraps cleanly. Model must fit on a
   single GPU in this mode.
"""

# --- header guard: must run before importing torch / evo2 / biofoundation ---
# From biofoundation/examples/evo2_llr.py. When launched via torchrun, each
# rank pins to its assigned GPU so HF Trainer sees only one device per rank
# (preventing DP wrapping) and Evo2/Vortex on that rank only uses that GPU.
import os

_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank))
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

from bolinas.evals.evo2 import compute_evo2_llr, scores_dataframe  # noqa: E402


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
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device eval batch size. If omitted, auto-tune "
                        "by OOM-descent from --tune-start.")
    p.add_argument("--tune-start", type=int, default=64,
                   help="Starting batch size for OOM-descent tuning.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Only score the first N variants. Smoke-test flag.")
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/evo2_traitgym_v2/{args.model}_{args.split}.parquet"

    ds = load_dataset(DATASET_HF_PATH, split=args.split).to_pandas()
    missing = [c for c in REQUIRED_COLS if c not in ds.columns]
    assert not missing, f"dataset is missing required columns: {missing}"
    assert ds["label"].isna().sum() == 0, "label column contains NaN"
    assert set(ds["label"].unique()) <= {0, 1}, "label is not binary 0/1"
    if args.limit is not None:
        ds = ds.head(args.limit).reset_index(drop=True)
        print(f"[evo2] --limit {args.limit} applied, scoring {len(ds)} variants")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[evo2] model={args.model} split={args.split} n={len(ds)}")
        print(f"[evo2] subsets:\n{ds['subset'].value_counts().to_string()}")

    llr = compute_evo2_llr(
        model_name=args.model,
        dataset=ds[["chrom", "pos", "ref", "alt"]],
        genome_path=args.genome_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        tune_start=args.tune_start,
        num_workers=args.num_workers,
    )

    # When launched via torchrun with DDP, HF Trainer gathers predictions on
    # every rank. Only rank 0 needs to write the parquet.
    if rank != 0:
        return

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

"""Sanity check: compare our saved LLR against evo2's own score_sequences delta.

Reproduces the approach from biofoundation/examples/test_evo2.py on the first N
variants of TraitGym Mendelian v2 train split, then diffs against the LLR we
stored in results/evo2_traitgym_v2/{model}_train.parquet.

If the two match (within fp8 noise), our eval pipeline is sign- and
construction-correct.
"""

import os

# --- header guard copied from biofoundation/examples/evo2_llr.py ---
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_local_rank))
# -------------------------------------------------------------------

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from biofoundation.data import Genome, _get_variant_window  # noqa: E402
from datasets import load_dataset  # noqa: E402
from evo2 import Evo2  # noqa: E402


DATASET_HF_PATH = "bolinas-dna/evals-traitgym_mendelian_v2"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="evo2_1b_base")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--genome-path", default="results/genome.fa.gz")
    p.add_argument(
        "--parquet",
        default=None,
        help="Path to the LLR parquet to compare against (defaults to "
        "results/evo2_traitgym_v2/{model}_{split}.parquet).",
    )
    args = p.parse_args()
    if args.parquet is None:
        args.parquet = f"results/evo2_traitgym_v2/{args.model}_{args.split}.parquet"

    # Load the first n variants, exactly as our pipeline would have seen them.
    ds = load_dataset(DATASET_HF_PATH, split=args.split).to_pandas().head(args.n)
    print(f"loaded first {len(ds)} variants from {DATASET_HF_PATH} / {args.split}")

    genome = Genome(args.genome_path)
    model = Evo2(args.model)

    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    for _, v in ds.iterrows():
        seq, pos = _get_variant_window(v.to_dict(), genome, args.window_size)
        ref_seqs.append(seq)
        alt_seqs.append(seq[:pos] + v["alt"] + seq[pos + 1 :])
        assert seq[pos] == v["ref"], (
            f"ref mismatch at {v.chrom}:{v.pos}: genome={seq[pos]!r} vs dataset ref={v['ref']!r}"
        )

    print(f"scoring {len(ref_seqs)} ref sequences...")
    ref_scores = np.array(model.score_sequences(ref_seqs, reduce_method="sum"))
    print(f"scoring {len(alt_seqs)} alt sequences...")
    alt_scores = np.array(model.score_sequences(alt_seqs, reduce_method="sum"))

    reference_llr = alt_scores - ref_scores  # matches biofoundation convention

    # Pull the equivalent rows out of our parquet. We keyed on (chrom,pos,ref,alt)
    # in the full run; since ds is head(n), matching rows are the first n in the
    # parquet too, but verify by join to be paranoid.
    stored = pd.read_parquet(args.parquet).head(args.n)
    for col in ["chrom", "pos", "ref", "alt"]:
        assert (stored[col].values == ds[col].values).all(), (
            f"row order mismatch on column {col}"
        )
    stored_llr = stored["llr"].values.astype(float)

    print("\n--- comparison (first 10) ---")
    print(f"{'ref_llr':>12}  {'stored':>12}  {'diff':>12}  {'rel':>10}")
    max_abs = 0.0
    for i, (r, s) in enumerate(zip(reference_llr, stored_llr)):
        diff = float(r - s)
        rel = float(abs(diff) / max(abs(r), abs(s), 1e-6))
        max_abs = max(max_abs, abs(diff))
        print(f"{r:>12.4f}  {s:>12.4f}  {diff:>+12.4f}  {rel:>10.4%}")
    print(f"\nmax |diff| = {max_abs:.6f}")

    # Pass threshold: fp8 adds significant noise; anything <0.5 in absolute LLR
    # units is a reasonable "matches within fp8 + numerical noise" bar.
    ok = max_abs < 0.5
    print(f"{'PASS' if ok else 'FAIL'}: max_abs_diff {'<' if ok else '>='} 0.5")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

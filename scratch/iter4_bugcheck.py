"""Bug-check: is exp55-mammals RC-symmetric at the variant-position logit level?

For each of N random 5'UTR variants:
  - Run FWD forward pass on the 4 ACGT candidates of the FWD window
  - Run RC  forward pass on the 4 ACGT candidates of the RC window
  - Extract the LOGIT at the variant position from a single pass (ref candidate)
    on both sides, softmax-normalize to get P(A,C,G,T at variant pos).
  - For RC, complement the labels: P_rc(A) at variant pos → corresponds biologically
    to P(T) at the FWD variant pos.
  - Check whether P_fwd ≈ permuted(P_rc) per variant.

If exp55 was trained with RC augmentation and my implementation is correct,
these per-variant 4-vectors should agree very closely (correlation near 1).
If they disagree systematically, either the model isn't strand-symmetric or
my RC implementation has a bug somewhere I haven't spotted.

NOTE: this is GPU-bound, so run on the SkyPilot cluster, not locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biofoundation.data import Genome
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.zeroshot_vep.features import _get_centered_window, _get_special_token_counts
from bolinas.zeroshot_vep.scores import NUC_TO_IDX, NUCLEOTIDES


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT[c] for c in seq[::-1])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-per-subset", type=int, default=10,
                    help="Variants to sample per subset (stratified).")
    ap.add_argument("--subsets", type=str,
                    default="5_prime_UTR_variant,missense_variant,synonymous_variant,splicing",
                    help="Comma-separated list of subset names to stratify across.")
    ap.add_argument("--source-parquet", type=str,
                    default="scratch/iter1/scores/exp55-mammals__win256__mendelian_traits.parquet",
                    help="Where to read the variant list from (just need chrom/pos/ref/alt/subset).")
    ap.add_argument("--out", default="scratch/iter4/iter4_bugcheck.parquet")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    # Use any iter-1 mendelian parquet to keep variants the same.
    src = pd.read_parquet(args.source_parquet)
    # Stratified sample.
    pool = pd.concat([
        src[src["subset"] == sub].sample(min(args.n_per_subset, (src["subset"] == sub).sum()), random_state=0)
        for sub in subsets
    ]).reset_index(drop=True)
    print(f"[bugcheck] {len(pool)} variants picked across subsets {subsets}", flush=True)
    print(f"[bugcheck] per-subset counts:\n{pool['subset'].value_counts().to_string()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    genome = Genome(args.genome)
    if hasattr(genome, "_genome"):
        genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0
    W = args.window
    fwd_var_idx = W // 2
    rc_var_idx = W - 1 - fwd_var_idx
    fwd_tok_var_pos = fwd_var_idx + n_prefix
    rc_tok_var_pos = rc_var_idx + n_prefix
    print(f"[bugcheck] W={W} fwd_var_idx={fwd_var_idx} rc_var_idx={rc_var_idx} "
          f"fwd_tok_pos={fwd_tok_var_pos} rc_tok_pos={rc_tok_var_pos}", flush=True)

    # For a CLM, the logit at position p-1 predicts the token at position p.
    # So to score the variant token, we read logits at (tok_var_pos - 1).
    fwd_predict_idx = fwd_tok_var_pos - 1
    rc_predict_idx = rc_tok_var_pos - 1

    # Token IDs for A/C/G/T (so we can pick logits[predict_idx, [tA,tC,tG,tT]]).
    nuc_token_ids = [tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES]
    print(f"[bugcheck] nuc_token_ids = {dict(zip(NUCLEOTIDES, nuc_token_ids))}", flush=True)

    records = []
    for i, (_, row) in enumerate(pool.iterrows()):
        # FWD pass: ref-candidate window (just one pass, we want logits at variant_pos-1)
        seq, _ = _get_centered_window(genome, row["chrom"], int(row["pos"]), row["ref"], W)
        assert seq[fwd_var_idx] == row["ref"]
        # Use the ref-allele sequence for the diagnostic; alt would give similar logits since
        # we're reading position p-1 which doesn't include the variant token.
        fwd_ids = torch.tensor(tokenizer.encode(seq), device=device).unsqueeze(0)
        rc_seq = revcomp(seq)
        assert rc_seq[rc_var_idx] == COMPLEMENT[row["ref"]]
        rc_ids = torch.tensor(tokenizer.encode(rc_seq), device=device).unsqueeze(0)

        with torch.no_grad():
            fwd_logits = model(fwd_ids).logits[0, fwd_predict_idx].float()  # (V,)
            rc_logits = model(rc_ids).logits[0, rc_predict_idx].float()  # (V,)

        # Restrict to A/C/G/T and softmax.
        fwd_logp = F.log_softmax(fwd_logits[nuc_token_ids], dim=-1).cpu().numpy()
        rc_logp = F.log_softmax(rc_logits[nuc_token_ids], dim=-1).cpu().numpy()
        # If RC-symmetric: P_fwd(nuc) should equal P_rc(complement(nuc)).
        # Reorder rc_logp so it's in [A_fwd, C_fwd, G_fwd, T_fwd] order:
        #   complement(A) = T → rc_logp[T_idx]; complement(C) = G → rc_logp[G_idx]; etc.
        rc_logp_complemented = np.array([
            rc_logp[NUC_TO_IDX[COMPLEMENT[nuc]]] for nuc in NUCLEOTIDES
        ])

        records.append({
            "chrom": row["chrom"], "pos": int(row["pos"]),
            "ref": row["ref"], "alt": row["alt"],
            "subset": row["subset"], "label": int(row["label"]),
            **{f"fwd_logp_{nuc}": float(fwd_logp[i]) for i, nuc in enumerate(NUCLEOTIDES)},
            **{f"rc_logp_compl_{nuc}": float(rc_logp_complemented[i]) for i, nuc in enumerate(NUCLEOTIDES)},
        })
        if (i + 1) % 5 == 0:
            print(f"[bugcheck]   {i+1}/{len(pool)}", flush=True)

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[bugcheck] wrote {args.out}", flush=True)

    # Quick summary inline.
    print("\n=== Per-variant logp(nuc): FWD vs RC-complemented ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # For each variant compute per-variant Pearson r between fwd_logp[A,C,G,T]
    # and rc_logp_compl[A,C,G,T], and per-variant MSE.
    rs = []
    mses = []
    for _, r in df.iterrows():
        f = np.array([r[f"fwd_logp_{nuc}"] for nuc in NUCLEOTIDES])
        rc = np.array([r[f"rc_logp_compl_{nuc}"] for nuc in NUCLEOTIDES])
        if np.std(f) > 0 and np.std(rc) > 0:
            r_val = np.corrcoef(f, rc)[0, 1]
        else:
            r_val = np.nan
        rs.append(r_val)
        mses.append(float(np.mean((f - rc) ** 2)))
    df["pearson_r_fwd_vs_rc_compl"] = rs
    df["mse_fwd_vs_rc_compl"] = mses
    # Re-write parquet with the per-variant diagnostics.
    df.to_parquet(args.out, index=False)

    print("\n=== Overall: Pearson FWD-vs-RC-complemented ===")
    rs = np.array(rs)
    print(f"median = {np.nanmedian(rs):.3f}, mean = {np.nanmean(rs):.3f}, "
          f"min = {np.nanmin(rs):.3f}, max = {np.nanmax(rs):.3f}")

    print("\n=== Per-subset summary (median Pearson, median MSE) ===")
    by_subset = df.groupby("subset").agg(
        n=("chrom", "size"),
        median_pearson=("pearson_r_fwd_vs_rc_compl", "median"),
        mean_pearson=("pearson_r_fwd_vs_rc_compl", "mean"),
        median_mse=("mse_fwd_vs_rc_compl", "median"),
        mean_mse=("mse_fwd_vs_rc_compl", "mean"),
    ).reset_index()
    print(by_subset.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Also: max absolute deviation between FWD and RC-complemented logp anywhere
    print("\n=== Max |FWD - RC-complemented| logp across variants × nucleotides ===")
    max_dev = max(
        max(abs(r[f"fwd_logp_{nuc}"] - r[f"rc_logp_compl_{nuc}"]) for nuc in NUCLEOTIDES)
        for _, r in df.iterrows()
    )
    print(f"max |Δlogp| = {max_dev:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

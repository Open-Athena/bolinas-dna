"""Iter 3 scout: downstream-effect scores from a single model on all 3 datasets.

Per variant, run 2 forward passes (REF-context, ALT-context), get the
4-nucleotide softmax at each output position, and compute 8 candidate scores
that quantify how the alt allele perturbs the model's downstream predictions:

  range = output indices p, p+1, ..., T-2 (immediate-affected through end)
  metrics on the per-position 4-prob distribution:
    - jsd:  ½·KL(p_ref ‖ m) + ½·KL(p_alt ‖ m)  where m = (p_ref + p_alt)/2
    - l1:   Σ_k |p_ref(k) - p_alt(k)|
    - l2:   √Σ_k (p_ref(k) - p_alt(k))²
    - linf: max_k |p_ref(k) - p_alt(k)|
  aggregations across positions in the range:
    - mean
    - max
  → 4 × 2 = 8 scores per variant

Configured for exp55-mammals × win=256 on all 3 datasets (full, no subset
filter — analysis script restricts to tss_proximal + 5_prime_UTR_variant
since exp55-mammals is the promoter model).

Per-variant softmax tensors are large but ephemeral — computed in-batch and
collapsed to the 8 scalar scores before the next batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biofoundation.data import Genome
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.zeroshot_vep.features import (
    _get_centered_window,
    _get_special_token_counts,
)
from bolinas.zeroshot_vep.scores import NUCLEOTIDES


def _get_nucleotide_token_ids(tokenizer) -> dict[str, int]:
    """Single-token ID for each of {A, C, G, T} under this tokenizer."""
    n_prefix, _ = _get_special_token_counts(tokenizer)
    return {nuc: tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES}


SCORE_COLS = [
    "down_jsd_mean", "down_jsd_max",
    "down_l1_mean",  "down_l1_max",
    "down_l2_mean",  "down_l2_max",
    "down_linf_mean","down_linf_max",
]


def _slice_alphabet_logprobs(
    log_softmax: torch.Tensor, alphabet_ids: torch.Tensor
) -> torch.Tensor:
    """Pick out the 4 nucleotide entries from a (B, T, V) log_softmax tensor.

    Returns shape (B, T, 4) — log-prob of each of {A, C, G, T} per position.
    """
    # Gather along vocab dim. alphabet_ids: shape (4,)
    return log_softmax.index_select(dim=-1, index=alphabet_ids)


def _renormalize_4(logprobs_4: torch.Tensor) -> torch.Tensor:
    """Re-normalize a (..., 4) log-prob slice so the 4 nucleotides sum to 1.

    The model's softmax is over the full vocab; restricting to {A,C,G,T} gives
    a sub-distribution that doesn't sum to 1. For distance metrics on the
    "nucleotide-only" distribution, we need to re-normalize.
    """
    return logprobs_4 - torch.logsumexp(logprobs_4, dim=-1, keepdim=True)


def _jsd(p_ref: torch.Tensor, p_alt: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """JSD between two probability vectors along the last axis.

    Args:
        p_ref, p_alt: (..., K) probability vectors, summing to 1.

    Returns:
        (...,) tensor of JSD values in nats, bounded in [0, log 2].
    """
    m = 0.5 * (p_ref + p_alt)
    # KL(p || m) = Σ p log(p/m); use eps to avoid log(0).
    def _kl(p, q):
        return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    return 0.5 * (_kl(p_ref, m) + _kl(p_alt, m))


def compute_downstream_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    window_size: int,
    batch_size: int = 32,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> pd.DataFrame:
    """Compute the 8 downstream-effect scores per variant via 2 forward passes.

    Returns a DataFrame of shape (N, 8) row-aligned with the input dataset.
    """
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    genome = Genome(genome_path)
    # Speedup: biofoundation.data.Genome stores chromosome sequences in a
    # pandas Series, and recent pyarrow-backed Series make per-call lookup
    # slow (py-spy showed Genome.__call__ dominating CPU time). Replace with
    # a plain dict for O(1) python access. Same string content; same slicing.
    if hasattr(genome, "_genome"):
        genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}
        print(f"[iter3] converted Genome backing to dict ({len(genome._genome)} chroms)",
              flush=True)
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0

    # Token IDs for the 4 nucleotides — used to slice the softmax.
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    alphabet_ids = torch.tensor(
        [nuc_ids[nuc] for nuc in NUCLEOTIDES], dtype=torch.long, device=device
    )

    probe_row = dataset.iloc[0]
    probe_seq, dna_var_idx = _get_centered_window(
        genome, probe_row["chrom"], int(probe_row["pos"]),
        probe_row["ref"], window_size,
    )
    probe_ids = torch.tensor(tokenizer.encode(probe_seq), device=device)
    T_tok = probe_ids.shape[0]
    tok_var_pos = dna_var_idx + n_prefix  # input position of the variant
    print(
        f"[iter3] checkpoint={checkpoint_path.name} window={window_size} "
        f"T_tok={T_tok} tok_var_pos={tok_var_pos} dtype={dtype}",
        flush=True,
    )

    # Range: output indices i ≥ tok_var_pos. Output index i is the prediction
    # for input position i+1, conditioning on input positions 0..i. Output at
    # i=tok_var_pos is the first that sees the variant in its context.
    # Output tensor shape: (B, T, V); we restrict to (B, tok_var_pos:T, V).
    # That's the slice that has signal — output indices [tok_var_pos, T-1].
    # Note: output at index T-1 predicts a "phantom" position past the end and
    # is uninformative; include it for completeness and trust the metrics to
    # be small there (or skip — for consistency we keep all i >= tok_var_pos
    # in the slice and don't worry).

    N = len(dataset)
    out = np.empty((N, len(SCORE_COLS)), dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_rows = dataset.iloc[batch_start:batch_end]
        B = len(batch_rows)

        # Build (B, 2, T) input_ids: index 0 = REF context, index 1 = ALT context.
        batch_inputs: list[torch.Tensor] = []
        for _, row in batch_rows.iterrows():
            seq, var_idx = _get_centered_window(
                genome, row["chrom"], int(row["pos"]), row["ref"], window_size
            )
            ref_seq = seq
            alt_seq = seq[:var_idx] + row["alt"] + seq[var_idx + 1 :]
            pair = torch.stack([
                torch.tensor(tokenizer.encode(ref_seq), dtype=torch.long),
                torch.tensor(tokenizer.encode(alt_seq), dtype=torch.long),
            ])
            batch_inputs.append(pair)
        input_ids = torch.stack(batch_inputs).to(device)  # (B, 2, T)
        input_ids_flat = input_ids.reshape(B * 2, T_tok)

        with torch.no_grad():
            logits = model(input_ids_flat).logits  # (B*2, T, V)
        # log_softmax in fp32 for numerical stability.
        log_p_full = F.log_softmax(logits.float(), dim=-1)  # (B*2, T, V)
        # Slice out the 4 nucleotide entries → (B*2, T, 4).
        log_p_4 = _slice_alphabet_logprobs(log_p_full, alphabet_ids)
        # Re-normalize so the 4 probs sum to 1.
        log_p_4 = _renormalize_4(log_p_4)
        # Reshape: (B, 2, T, 4)
        log_p_4 = log_p_4.reshape(B, 2, T_tok, 4)
        # Slice to the range of interest: output indices i >= tok_var_pos.
        # In tensor space, that's positions tok_var_pos..T-1 of the T axis.
        log_p_4 = log_p_4[:, :, tok_var_pos:, :]  # (B, 2, T_eff, 4)
        # Convert to probs.
        p_4 = log_p_4.exp()
        p_ref = p_4[:, 0]  # (B, T_eff, 4)
        p_alt = p_4[:, 1]

        # Per-position metrics (B, T_eff).
        diff = p_ref - p_alt
        l1 = diff.abs().sum(dim=-1)                       # (B, T_eff)
        l2 = (diff * diff).sum(dim=-1).sqrt()
        linf = diff.abs().max(dim=-1).values
        jsd = _jsd(p_ref, p_alt)

        # Aggregations across positions: mean + max.
        def agg(t: torch.Tensor) -> tuple[float, float]:
            return t.mean(dim=-1), t.max(dim=-1).values

        agg_results = {
            "down_jsd_mean":  jsd.mean(dim=-1),  "down_jsd_max":  jsd.max(dim=-1).values,
            "down_l1_mean":   l1.mean(dim=-1),   "down_l1_max":   l1.max(dim=-1).values,
            "down_l2_mean":   l2.mean(dim=-1),   "down_l2_max":   l2.max(dim=-1).values,
            "down_linf_mean": linf.mean(dim=-1), "down_linf_max": linf.max(dim=-1).values,
        }

        for j, col in enumerate(SCORE_COLS):
            out[batch_start:batch_end, j] = agg_results[col].float().cpu().numpy()

        if batch_start % (batch_size * 20) == 0:
            print(f"[iter3]   {batch_end}/{N} variants", flush=True)

    return pd.DataFrame(out, columns=SCORE_COLS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to HF checkpoint dir")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--datasets", nargs="+",
                    default=["mendelian_traits", "complex_traits", "eqtl"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-hf-prefix", default="bolinas-dna/evals")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        hf_path = f"{args.input_hf_prefix}_{dataset_name}"
        print(f"\n[iter3] loading {hf_path} split={args.split}", flush=True)
        ds = load_dataset(hf_path, split=args.split).to_pandas()
        print(f"[iter3] {dataset_name}: {len(ds)} variants, subsets={sorted(ds['subset'].unique())}",
              flush=True)

        scores_df = compute_downstream_scores(
            checkpoint_path=args.checkpoint,
            dataset=ds,
            genome_path=args.genome,
            window_size=args.window,
            batch_size=args.batch_size,
        )
        # Concat with variant metadata.
        meta_cols = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
        out = pd.concat([ds[meta_cols].reset_index(drop=True), scores_df], axis=1)

        out_path = out_dir / f"iter3_exp55-mammals__win{args.window}__{dataset_name}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"[iter3] wrote {out_path} ({len(out)} rows × {scores_df.shape[1]} scores)",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Iter-6 scout: downstream-effect (nucleotide-dependency) scores with strand control.

Same 8 scores as iter-3 (`down_{jsd,l1,l2,linf}_{mean,max}`) but parameterized
over ``--strand fwd|rc`` so we can run both strands separately and average them
on the CPU side (matching the iter-5 RC AVG protocol). Adds ``--checkpoint`` /
``--dataset`` / ``--out`` so we can drive it from a sky yaml the same way as
the iter-4 ``zeroshot_vep_iter4_precision_scout.py`` is driven.

Per variant, 2 forward passes (REF-context, ALT-context). For ``--strand rc``,
the centered window is reverse-complemented and the variant's
ref/alt are complemented before being substituted into the RC window. The
"downstream" range in tokenizer space is the same in both cases (output indices
``≥ tok_var_pos``); semantically, FWD captures the genomic-downstream half and
RC captures the genomic-upstream half. AVG therefore probes the *bidirectional*
nucleotide-dependency footprint that the AR model could only emit unilaterally
per strand.
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


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT[c] for c in seq[::-1])


def _get_nucleotide_token_ids(tokenizer) -> dict[str, int]:
    n_prefix, _ = _get_special_token_counts(tokenizer)
    return {nuc: tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES}


SCORE_COLS = [
    "down_jsd_mean", "down_jsd_max",
    "down_l1_mean",  "down_l1_max",
    "down_l2_mean",  "down_l2_max",
    "down_linf_mean","down_linf_max",
]


def _slice_alphabet_logprobs(log_softmax: torch.Tensor, alphabet_ids: torch.Tensor) -> torch.Tensor:
    return log_softmax.index_select(dim=-1, index=alphabet_ids)


def _renormalize_4(logprobs_4: torch.Tensor) -> torch.Tensor:
    return logprobs_4 - torch.logsumexp(logprobs_4, dim=-1, keepdim=True)


def _jsd(p_ref: torch.Tensor, p_alt: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    m = 0.5 * (p_ref + p_alt)
    def _kl(p, q):
        return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    return 0.5 * (_kl(p_ref, m) + _kl(p_alt, m))


def compute_downstream_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    window_size: int,
    strand: str,
    batch_size: int = 16,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> pd.DataFrame:
    """Compute 8 downstream-effect scores per variant on either FWD or RC strand."""
    assert strand in ("fwd", "rc"), strand

    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    genome = Genome(genome_path)
    if hasattr(genome, "_genome"):
        genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}
        print(f"[iter6] converted Genome backing to dict ({len(genome._genome)} chroms)", flush=True)

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0
    W = window_size

    # Variant index in the (possibly RC'd) DNA window (matches iter-4 precision_scout).
    if strand == "fwd":
        dna_var_idx = W // 2
    else:
        dna_var_idx = W - 1 - (W // 2)
    tok_var_pos = dna_var_idx + n_prefix

    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    alphabet_ids = torch.tensor(
        [nuc_ids[nuc] for nuc in NUCLEOTIDES], dtype=torch.long, device=device
    )

    # Probe to fix tokenized window length.
    probe_row = dataset.iloc[0]
    probe_seq, _ = _get_centered_window(
        genome, probe_row["chrom"], int(probe_row["pos"]),
        probe_row["ref"], W,
    )
    if strand == "rc":
        probe_seq = revcomp(probe_seq)
    probe_ids = torch.tensor(tokenizer.encode(probe_seq), device=device)
    T_tok = probe_ids.shape[0]
    print(
        f"[iter6] strand={strand} checkpoint={checkpoint_path.name} window={W} "
        f"T_tok={T_tok} tok_var_pos={tok_var_pos} dtype={dtype}",
        flush=True,
    )

    N = len(dataset)
    out = np.empty((N, len(SCORE_COLS)), dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_rows = dataset.iloc[batch_start:batch_end]
        B = len(batch_rows)

        batch_inputs: list[torch.Tensor] = []
        for _, row in batch_rows.iterrows():
            seq, var_idx = _get_centered_window(
                genome, row["chrom"], int(row["pos"]), row["ref"], W
            )
            if strand == "fwd":
                ref_nuc = row["ref"]
                alt_nuc = row["alt"]
                # var_idx returned by _get_centered_window is the FWD-window position.
                ref_seq = seq
                alt_seq = seq[:var_idx] + alt_nuc + seq[var_idx + 1:]
            else:
                rc_seq = revcomp(seq)
                ref_nuc = COMPLEMENT[row["ref"]]
                alt_nuc = COMPLEMENT[row["alt"]]
                ref_seq = rc_seq[:dna_var_idx] + ref_nuc + rc_seq[dna_var_idx + 1:]
                alt_seq = rc_seq[:dna_var_idx] + alt_nuc + rc_seq[dna_var_idx + 1:]
            pair = torch.stack([
                torch.tensor(tokenizer.encode(ref_seq), dtype=torch.long),
                torch.tensor(tokenizer.encode(alt_seq), dtype=torch.long),
            ])
            batch_inputs.append(pair)
        input_ids = torch.stack(batch_inputs).to(device)  # (B, 2, T)
        input_ids_flat = input_ids.reshape(B * 2, T_tok)

        with torch.no_grad():
            logits = model(input_ids_flat).logits  # (B*2, T, V)
        log_p_full = F.log_softmax(logits.float(), dim=-1)
        log_p_4 = _slice_alphabet_logprobs(log_p_full, alphabet_ids)
        log_p_4 = _renormalize_4(log_p_4)
        log_p_4 = log_p_4.reshape(B, 2, T_tok, 4)
        log_p_4 = log_p_4[:, :, tok_var_pos:, :]
        p_4 = log_p_4.exp()
        p_ref = p_4[:, 0]
        p_alt = p_4[:, 1]

        diff = p_ref - p_alt
        l1 = diff.abs().sum(dim=-1)
        l2 = (diff * diff).sum(dim=-1).sqrt()
        linf = diff.abs().max(dim=-1).values
        jsd = _jsd(p_ref, p_alt)

        agg_results = {
            "down_jsd_mean":  jsd.mean(dim=-1),  "down_jsd_max":  jsd.max(dim=-1).values,
            "down_l1_mean":   l1.mean(dim=-1),   "down_l1_max":   l1.max(dim=-1).values,
            "down_l2_mean":   l2.mean(dim=-1),   "down_l2_max":   l2.max(dim=-1).values,
            "down_linf_mean": linf.mean(dim=-1), "down_linf_max": linf.max(dim=-1).values,
        }
        for j, col in enumerate(SCORE_COLS):
            out[batch_start:batch_end, j] = agg_results[col].float().cpu().numpy()

        if batch_start % (batch_size * 20) == 0:
            print(f"[iter6]   {batch_end}/{N} variants", flush=True)

    return pd.DataFrame(out, columns=SCORE_COLS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--window", type=int, default=255)
    ap.add_argument("--dataset", required=True,
                    help="One of mendelian_traits, complex_traits, eqtl.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-hf-prefix", default="bolinas-dna/evals")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--strand", choices=("fwd", "rc"), required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    hf_path = f"{args.input_hf_prefix}_{args.dataset}"
    print(f"[iter6] loading {hf_path} split={args.split}", flush=True)
    ds = load_dataset(hf_path, split=args.split).to_pandas()
    print(f"[iter6] {args.dataset}: {len(ds)} variants", flush=True)

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    scores_df = compute_downstream_scores(
        checkpoint_path=args.checkpoint,
        dataset=ds,
        genome_path=args.genome,
        window_size=args.window,
        strand=args.strand,
        batch_size=args.batch_size,
        dtype=torch_dtype,
    )
    meta_cols = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    out = pd.concat([ds[meta_cols].reset_index(drop=True), scores_df], axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[iter6] wrote {out_path} ({len(out)} rows × {scores_df.shape[1]} scores)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Iter-4 precision test: same 30 scores at fp32 model vs bf16 model.

Compute FWD or RC strand scores on a chosen model at a chosen dtype, so we
can compare bf16 (iter-1 baseline) to fp32 and see whether bf16 numerical
noise is a meaningful contributor to the FWD-vs-RC redundancy we observed
on exp58.

Identical logic to ``zeroshot_vep_iter4_rc_scout.py`` but parameterized over:
- ``--strand fwd|rc``: which strand to score
- ``--dtype bf16|fp32``: model forward-pass precision (log-softmax is always fp32)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biofoundation.data import Genome
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.zeroshot_vep.features import (
    _get_centered_window,
    _get_special_token_counts,
    _per_position_logprobs,
)
from bolinas.zeroshot_vep.scores import (
    NUC_TO_IDX,
    NUCLEOTIDES,
    SCORE_NAMES,
    embedding_scores,
    likelihood_scores,
)


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT[c] for c in seq[::-1])


def _build_candidates(seq: str, var_idx: int) -> list[str]:
    return [seq[:var_idx] + nuc + seq[var_idx + 1 :] for nuc in NUCLEOTIDES]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--dataset", default="mendelian_traits")
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-hf-prefix", default="bolinas-dna/evals")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--strand", choices=("fwd", "rc"), required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), required=True)
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[precision] strand={args.strand} dtype={args.dtype} window={args.window}", flush=True)

    print(f"[precision] loading {args.input_hf_prefix}_{args.dataset} split={args.split}", flush=True)
    ds = load_dataset(f"{args.input_hf_prefix}_{args.dataset}", split=args.split).to_pandas()
    print(f"[precision] {args.dataset}: {len(ds)} variants", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    genome = Genome(args.genome)
    if hasattr(genome, "_genome"):
        genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}
        print(f"[precision] converted Genome backing to dict ({len(genome._genome)} chroms)", flush=True)

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0
    W = args.window

    # Variant index in the (possibly RC'd) DNA window.
    if args.strand == "fwd":
        dna_var_idx = W // 2
    else:
        dna_var_idx = W - 1 - (W // 2)
    tok_var_pos = dna_var_idx + n_prefix
    print(f"[precision] dna_var_idx={dna_var_idx} tok_var_pos={tok_var_pos}", flush=True)

    # Probe T and D.
    probe_row = ds.iloc[0]
    probe_seq, _ = _get_centered_window(
        genome, probe_row["chrom"], int(probe_row["pos"]), probe_row["ref"], W
    )
    if args.strand == "rc":
        probe_seq = revcomp(probe_seq)
    probe_ids = torch.tensor(tokenizer.encode(probe_seq), device=device)
    T_tok = probe_ids.shape[0]
    print(f"[precision] T_tok={T_tok}", flush=True)

    with torch.no_grad():
        out = model(probe_ids.unsqueeze(0), output_hidden_states=True)
        n_hidden_states = len(out.hidden_states)
        D = out.hidden_states[-1].shape[-1]
        middle_idx = n_hidden_states // 2

    N = len(ds)
    score_arr = np.empty((N, len(SCORE_NAMES)), dtype=np.float32)
    batch_size = args.batch_size
    b_arange = torch.arange(batch_size, device=device)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_rows = ds.iloc[batch_start:batch_end]
        B = len(batch_rows)

        batch_input_ids: list[torch.Tensor] = []
        batch_ref_idx: list[int] = []
        batch_alt_idx: list[int] = []
        for _, row in batch_rows.iterrows():
            seq, _ = _get_centered_window(genome, row["chrom"], int(row["pos"]), row["ref"], W)
            if args.strand == "rc":
                seq = revcomp(seq)
                expected = COMPLEMENT[row["ref"]]
                ref_idx = NUC_TO_IDX[COMPLEMENT[row["ref"]]]
                alt_idx = NUC_TO_IDX[COMPLEMENT[row["alt"]]]
            else:
                expected = row["ref"]
                ref_idx = NUC_TO_IDX[row["ref"]]
                alt_idx = NUC_TO_IDX[row["alt"]]
            assert seq[dna_var_idx] == expected, (
                f"variant base mismatch: seq[{dna_var_idx}]={seq[dna_var_idx]} expected={expected}"
            )
            sequences = _build_candidates(seq, dna_var_idx)
            ids = torch.stack([
                torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in sequences
            ])
            batch_input_ids.append(ids)
            batch_ref_idx.append(ref_idx)
            batch_alt_idx.append(alt_idx)

        input_ids = torch.stack(batch_input_ids).to(device)
        input_ids_flat = input_ids.reshape(B * 4, T_tok)
        ref_idx_np = np.asarray(batch_ref_idx, dtype=np.int8)
        alt_idx_np = np.asarray(batch_alt_idx, dtype=np.int8)

        with torch.no_grad():
            model_out = model(input_ids_flat, output_hidden_states=True)
        logits = model_out.logits
        # Always cast to fp32 for log-softmax — biofoundation#21 lesson.
        pos_logp = _per_position_logprobs(logits.float(), input_ids_flat)
        seq_logp_b = pos_logp.reshape(B, 4, T_tok - 1).sum(dim=-1)
        del logits

        last_hidden_b = model_out.hidden_states[-1].reshape(B, 4, T_tok, D)
        middle_hidden_b = model_out.hidden_states[middle_idx].reshape(B, 4, T_tok, D)
        del model_out

        ref_idx_dev = torch.as_tensor(ref_idx_np, device=device, dtype=torch.long)
        alt_idx_dev = torch.as_tensor(alt_idx_np, device=device, dtype=torch.long)
        b_dev = b_arange[:B]
        emb_ref_last = last_hidden_b[b_dev, ref_idx_dev].to(torch.float16).cpu().numpy()
        emb_alt_last = last_hidden_b[b_dev, alt_idx_dev].to(torch.float16).cpu().numpy()
        emb_ref_mid = middle_hidden_b[b_dev, ref_idx_dev].to(torch.float16).cpu().numpy()
        emb_alt_mid = middle_hidden_b[b_dev, alt_idx_dev].to(torch.float16).cpu().numpy()
        del last_hidden_b, middle_hidden_b

        seq_logp_np = seq_logp_b.cpu().numpy()
        lik = likelihood_scores(seq_logp_np, ref_idx_np, alt_idx_np)
        emb = embedding_scores(
            emb_ref_last, emb_ref_mid, emb_alt_last, emb_alt_mid, var_pos=tok_var_pos,
        )
        all_scores = {**lik, **emb}
        for j, name in enumerate(SCORE_NAMES):
            score_arr[batch_start:batch_end, j] = all_scores[name]

        if batch_start % (batch_size * 50) == 0:
            print(f"[precision]   {batch_end}/{N} variants", flush=True)

    scores_df = pd.DataFrame(score_arr, columns=SCORE_NAMES)
    assert not scores_df.isna().any().any()

    out_meta = ds[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]].reset_index(drop=True)
    out_df = pd.concat([out_meta, scores_df], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[precision] wrote {args.out} ({len(out_df)} rows × {scores_df.shape[1]} scores)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

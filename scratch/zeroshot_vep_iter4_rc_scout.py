"""Iter 4 scout: reverse-complement-strand version of iter-1's 30 scores.

For each variant, run the same 4-pass forward inference as iter-1, but on
the REVERSE COMPLEMENT of the window. The variant's tokenized position in
RC space is ``W - 1 - (W // 2) + n_prefix`` (for even W, this differs from
FWD by 1; for odd W it's the same). Allele indices in RC space use the
complement: ``ref_idx_rc = NUC_TO_IDX[complement(ref)]``.

We compute the same 30 raw scores as iter-1 features.py (6 likelihood + 24
embedding) but on the RC strand. The forward-strand scores are already on
S3 from iter 1; downstream analysis merges FWD ↔ RC and adds AVG.

Scout: exp55-mammals × win=256 × mendelian_traits (train split) only.
"""

from __future__ import annotations

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


def _build_candidate_rc_sequences(rc_seq: str, var_idx_rc: int) -> list[str]:
    """4 candidate RC sequences with A/C/G/T at the RC-space variant position."""
    return [rc_seq[:var_idx_rc] + nuc + rc_seq[var_idx_rc + 1 :] for nuc in NUCLEOTIDES]


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
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args()

    print(f"[iter4_rc] loading {args.input_hf_prefix}_{args.dataset} split={args.split}", flush=True)
    ds = load_dataset(f"{args.input_hf_prefix}_{args.dataset}", split=args.split).to_pandas()
    print(f"[iter4_rc] {args.dataset}: {len(ds)} variants", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    genome = Genome(args.genome)
    # Genome-lookup-speed fix (per Open-Athena/biofoundation#23).
    if hasattr(genome, "_genome"):
        genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}
        print(f"[iter4_rc] converted Genome backing to dict ({len(genome._genome)} chroms)",
              flush=True)

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0
    W = args.window

    # FWD: variant at DNA index W // 2.
    # RC : variant at DNA index W - 1 - (W // 2).
    dna_var_idx_rc = W - 1 - (W // 2)
    print(f"[iter4_rc] W={W} dna_var_idx_fwd={W//2} dna_var_idx_rc={dna_var_idx_rc}", flush=True)

    # Probe T and D.
    probe_row = ds.iloc[0]
    probe_seq, _ = _get_centered_window(
        genome, probe_row["chrom"], int(probe_row["pos"]), probe_row["ref"], W
    )
    probe_rc = revcomp(probe_seq)
    probe_ids = torch.tensor(tokenizer.encode(probe_rc), device=device)
    T_tok = probe_ids.shape[0]
    tok_var_pos_rc = dna_var_idx_rc + n_prefix
    print(f"[iter4_rc] T_tok={T_tok} tok_var_pos_rc={tok_var_pos_rc}", flush=True)

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
        batch_ref_idx_rc: list[int] = []
        batch_alt_idx_rc: list[int] = []
        for _, row in batch_rows.iterrows():
            seq, _ = _get_centered_window(genome, row["chrom"], int(row["pos"]), row["ref"], W)
            rc_seq = revcomp(seq)
            # Sanity: the RC sequence at dna_var_idx_rc should be complement(ref).
            assert rc_seq[dna_var_idx_rc] == COMPLEMENT[row["ref"]], (
                f"RC variant base mismatch: rc[{dna_var_idx_rc}]={rc_seq[dna_var_idx_rc]} "
                f"complement(ref={row['ref']})={COMPLEMENT[row['ref']]} at {row['chrom']}:{row['pos']}"
            )
            sequences = _build_candidate_rc_sequences(rc_seq, dna_var_idx_rc)
            ids = torch.stack([
                torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in sequences
            ])  # (4, T_tok)
            batch_input_ids.append(ids)
            batch_ref_idx_rc.append(NUC_TO_IDX[COMPLEMENT[row["ref"]]])
            batch_alt_idx_rc.append(NUC_TO_IDX[COMPLEMENT[row["alt"]]])

        input_ids = torch.stack(batch_input_ids).to(device)  # (B, 4, T)
        input_ids_flat = input_ids.reshape(B * 4, T_tok)
        ref_idx_np = np.asarray(batch_ref_idx_rc, dtype=np.int8)
        alt_idx_np = np.asarray(batch_alt_idx_rc, dtype=np.int8)

        with torch.no_grad():
            model_out = model(input_ids_flat, output_hidden_states=True)
        logits = model_out.logits
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
            emb_ref_last, emb_ref_mid, emb_alt_last, emb_alt_mid,
            var_pos=tok_var_pos_rc,
        )
        all_scores = {**lik, **emb}
        for j, name in enumerate(SCORE_NAMES):
            score_arr[batch_start:batch_end, j] = all_scores[name]

        if batch_start % (batch_size * 50) == 0:
            print(f"[iter4_rc]   {batch_end}/{N} variants", flush=True)

    scores_df = pd.DataFrame(score_arr, columns=SCORE_NAMES)
    assert not scores_df.isna().any().any()

    out_meta = ds[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]].reset_index(drop=True)
    out_df = pd.concat([out_meta, scores_df], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[iter4_rc] wrote {args.out} ({len(out_df)} rows × {scores_df.shape[1]} scores)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

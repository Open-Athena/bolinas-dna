"""Zero-shot scoring variants probe for iter-2.

Goal: pick a LoRA score formula whose **zero-shot** PA is close to the
existing `embed_last_l2` baseline, while living in a canonical numerical
range so margin / lr / etc. are reusable from the metric-learning literature.

Variants computed per variant on the complex_traits train split:

1. `flat_l2`           — ||flat(last_ref) − flat(last_alt)||₂   (UNNORMALIZED, current baseline)
2. `flat_l2_norm`      — ||F.normalize(flat(last_ref)) − F.normalize(flat(last_alt))||₂  ∈ [0, 2]
3. `flat_cosine_dist`  — 1 − cos(flat(last_ref), flat(last_alt))                          ∈ [0, 2]  (monotone in 2)
4. `pool_l2`           — ||mean_pool(last_ref) − mean_pool(last_alt)||₂                   (UNNORMALIZED pool)
5. `pool_l2_norm`      — ||normalize(mean_pool(ref)) − normalize(mean_pool(alt))||₂        ∈ [0, 2]
6. `pool_cosine_dist`  — 1 − cos(mean_pool(ref), mean_pool(alt))                           ∈ [0, 2]
7. `per_pos_l2_sum`    — Σ_i ||last_ref[i] − last_alt[i]||₂                                (positional integral)
8. `per_pos_l2_max`    — max_i ||last_ref[i] − last_alt[i]||₂                              (positional max)

Run on cluster (GPU):
    cd ~/sky_workdir && ~/.local/bin/uv run --project . python -u scratch/iter2_zeroshot_variants.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biofoundation.data import Genome, transform_llr_clm
from biofoundation.model.adapters.hf import HFTokenizer
from datasets import load_dataset
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_DATASET = "bolinas-dna/evals_complex_traits"
SPLIT = "train"
BACKBONE = "bolinas-dna/exp166-p1B-step-16398"
WINDOW = 255
GENOME_PATH = (
    "/home/ubuntu/sky_workdir/snakemake/analysis/supervised_vep/.snakemake/storage/"
    "s3/oa-bolinas/snakemake/analysis/supervised_vep/results/genome.fa.gz"
)
BATCH_SIZE = 8
DEVICE = "cuda"


class _VariantDS(Dataset):
    def __init__(self, df, tokenizer, genome, window):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.genome = genome
        self.window = window

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        out = transform_llr_clm(
            {
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "ref": str(row["ref"]),
                "alt": str(row["alt"]),
            },
            tokenizer=self.tokenizer,
            genome=self.genome,
            window_size=self.window,
        )
        return out["input_ids"]  # [2, L]


@torch.no_grad()
def main():
    print(f"Loading dataset: {HF_DATASET}")
    df = load_dataset(HF_DATASET, split=SPLIT).to_pandas()
    print(f"  n={len(df)}")

    print(f"Loading backbone in bf16: {BACKBONE}")
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(BACKBONE))
    genome = Genome(Path(GENOME_PATH))

    ds = _VariantDS(df, tokenizer, genome, WINDOW)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=2)

    scores: dict[str, list[float]] = {
        k: []
        for k in (
            "flat_l2",
            "flat_l2_norm",
            "flat_cosine_dist",
            "pool_l2",
            "pool_l2_norm",
            "pool_cosine_dist",
            "per_pos_l2_sum",
            "per_pos_l2_max",
        )
    }

    t0 = time.time()
    for bi, batch in enumerate(loader):
        input_ids = batch.to(DEVICE)
        B = input_ids.shape[0]
        flat_ids = rearrange(input_ids, "B V L -> (B V) L")
        out = backbone(input_ids=flat_ids, output_hidden_states=True)
        last = out.hidden_states[-1].float()  # [B*2, L, D]
        last = rearrange(last, "(B V) L D -> B V L D", B=B)
        ref_lp = last[:, 0]  # [B, L, D]
        alt_lp = last[:, 1]
        ref_flat = rearrange(ref_lp, "B L D -> B (L D)")
        alt_flat = rearrange(alt_lp, "B L D -> B (L D)")

        scores["flat_l2"].extend(F.pairwise_distance(ref_flat, alt_flat).cpu().tolist())
        rn = F.normalize(ref_flat, dim=-1)
        an = F.normalize(alt_flat, dim=-1)
        scores["flat_l2_norm"].extend(F.pairwise_distance(rn, an).cpu().tolist())
        scores["flat_cosine_dist"].extend(
            (1.0 - F.cosine_similarity(ref_flat, alt_flat, dim=-1)).cpu().tolist()
        )

        rp = ref_lp.mean(dim=1)  # mean-pool over L → [B, D]
        ap = alt_lp.mean(dim=1)
        scores["pool_l2"].extend(F.pairwise_distance(rp, ap).cpu().tolist())
        rpn = F.normalize(rp, dim=-1)
        apn = F.normalize(ap, dim=-1)
        scores["pool_l2_norm"].extend(F.pairwise_distance(rpn, apn).cpu().tolist())
        scores["pool_cosine_dist"].extend(
            (1.0 - F.cosine_similarity(rp, ap, dim=-1)).cpu().tolist()
        )

        per_pos_l2 = (ref_lp - alt_lp).norm(dim=-1)  # [B, L]
        scores["per_pos_l2_sum"].extend(per_pos_l2.sum(dim=1).cpu().tolist())
        scores["per_pos_l2_max"].extend(per_pos_l2.max(dim=1).values.cpu().tolist())

        if (bi + 1) % 10 == 0:
            elapsed = time.time() - t0
            done = (bi + 1) * BATCH_SIZE
            print(f"  batch {bi + 1}/{len(loader)}  n_done={done}/{len(df)}  elapsed={elapsed:.1f}s")

    print(f"Forward pass done in {time.time() - t0:.1f}s")

    # Pack scores into a frame aligned with df.
    out_df = df.copy().reset_index(drop=True)
    for name, vals in scores.items():
        out_df[name] = np.array(vals[: len(out_df)], dtype=np.float64)

    # Save scored frame.
    out_path = Path("/home/ubuntu/iter2_zeroshot_variants.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved scored variants: {out_path} (n_rows={len(out_df)})")

    # Compute PairwiseAccuracy via the existing library code.
    from bolinas.evals.metrics import compute_pairwise_metrics

    score_cols = list(scores.keys())
    meta = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    m = compute_pairwise_metrics(
        dataset=out_df[meta], scores=out_df[score_cols], score_columns=score_cols
    )
    summary = m[m["subset"].isin(["_global_", "_macro_avg_"])][
        ["score_type", "subset", "value", "se", "n_pairs"]
    ].sort_values(["subset", "score_type"])
    print("\n========== ZERO-SHOT PA on complex_traits ==========")
    print(summary.to_string(index=False))
    summary.to_parquet("/home/ubuntu/iter2_zeroshot_variants_summary.parquet", index=False)


if __name__ == "__main__":
    main()

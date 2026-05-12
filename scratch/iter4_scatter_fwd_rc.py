"""FWD vs RC scatter plots for LLR per (model, subset).

Show distributional shape directly: identical (diagonal line), noisy-around-0
(small blob centered at origin), or "informative but disagreeing" (large blob
with low correlation).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bolinas.zeroshot_vep.scores import SCORE_NAMES, apply_score_directions


MODEL_SPECS = [
    ("exp55-mammals", 256),
    ("exp58-mammals", 256),
    ("exp59-mammals", 256),
    ("exp136-proj_v30", 255),
]

SUBSETS = ["3_prime_UTR_variant", "5_prime_UTR_variant", "distal", "missense_variant",
           "non_coding_transcript_exon_variant", "splicing", "synonymous_variant", "tss_proximal"]


def main() -> int:
    fig, axes = plt.subplots(len(MODEL_SPECS), len(SUBSETS), figsize=(28, 14), sharex=False, sharey=False)
    for i, (model, window) in enumerate(MODEL_SPECS):
        fp = Path(f"scratch/iter1/scores/{model}__win{window}__mendelian_traits.parquet")
        rp = Path(f"scratch/iter4/iter4_rc_{model}__win{window}__mendelian_traits.parquet")
        fwd = pd.read_parquet(fp)
        rc = pd.read_parquet(rp)
        key = ["chrom", "pos", "ref", "alt"]
        rc_renamed = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
        m = fwd.merge(rc_renamed, on=key)
        f_signed = apply_score_directions(m[SCORE_NAMES])
        rc_only = m[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
        r_signed = apply_score_directions(rc_only)

        for j, sub in enumerate(SUBSETS):
            ax = axes[i, j]
            sub_idx = m[m["subset"] == sub].index
            f_arr = f_signed.loc[sub_idx, "llr"].values
            r_arr = r_signed.loc[sub_idx, "llr"].values
            lim = max(np.percentile(np.abs(f_arr), 99), np.percentile(np.abs(r_arr), 99))
            ax.scatter(f_arr, r_arr, s=4, alpha=0.4)
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5, alpha=0.5)
            ax.axhline(0, color="gray", linewidth=0.3)
            ax.axvline(0, color="gray", linewidth=0.3)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("FWD LLR", fontsize=7)
            ax.set_ylabel("RC LLR", fontsize=7)
            ax.tick_params(labelsize=6)
            pearson = np.corrcoef(f_arr, r_arr)[0, 1]
            std_f, std_r = np.std(f_arr), np.std(r_arr)
            redundancy = np.mean((f_arr - r_arr) ** 2) / (np.var(f_arr) + np.var(r_arr))
            short_sub = sub.replace("_variant", "").replace("non_coding_transcript_exon", "ncRNA_exon")[:18]
            ax.set_title(f"{model}\n{short_sub} (n={len(f_arr)})\nr={pearson:.2f} std={std_f:.1f}/{std_r:.1f} red={redundancy:.2f}",
                        fontsize=8)
    plt.tight_layout()
    out = Path("scratch/iter4/iter4_fwd_rc_scatter_4models.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    main()

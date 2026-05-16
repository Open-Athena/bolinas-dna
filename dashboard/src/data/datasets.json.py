"""Observable Framework data loader: dataset-level metadata → JSON.

Per-dataset pins (HF commit, score type, tracking issue, etc.) for the
dataset-metadata block shown above the leaderboard.
"""

from __future__ import annotations

import json
import sys

DATASETS = {
    "mendelian_traits": {
        "name": "Mendelian Traits",
        "hf_repo": "bolinas-dna/evals_mendelian_traits",
        "hf_commit": "15e85bba",
        "score_type": "minus_llr",
        "leading_aggregate": "macro_avg",
        "issue": "https://github.com/Open-Athena/bolinas-dna/issues/161",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "OMIM ∪ HGMD ∪ Smedley et al. 2016 pathogenic SNVs (AF < 0.1%)",
        "negatives": "gnomAD high-frequency (AF > 0.1%)",
        "matching": "1:1 on gene, consequence, TSS distance, exon distance",
        "metric": "PairwiseAccuracy ± Wald SE — fraction of `(positive, negative)` pairs where the positive scores strictly higher than its matched negative (ties count 0.5). The `n` in each column header is the number of pairs.",
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`.",
            "Sorted by Macro Avg by default — the consequence-subset distribution (~92% missense) reflects human-annotator focus on protein-coding disease variants, not the underlying prevalence of pathogenic variants; Global PA therefore over-weights protein-coding-specialist methods. Macro Avg gives equal weight to each subset.",
        ],
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

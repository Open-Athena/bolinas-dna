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
        "positives": "HGMD ∪ OMIM ∪ Smedley et al. 2016 pathogenic SNVs (de-duped, AF<0.001)",
        "negatives": "gnomAD common-frequency variants (AN≥25k, AF>0.001)",
        "matching": "1:1 within gene + consequence subset",
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`.",
            "Sorted by Macro Avg by default — rationale: ~92% missense over-weights protein-coding-specialist methods on Global. Click any column header to re-sort.",
        ],
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

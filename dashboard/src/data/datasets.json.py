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
        "description": (
            "Mendelian disease pathogenic SNVs (HGMD ∪ OMIM ∪ Smedley et al. "
            "2016, de-duped, AF<0.001) × gnomAD common-frequency negatives "
            "(AN≥25k, AF>0.001), matched 1:1 within gene + consequence "
            "subset. Pathogenic should score higher than benign — hence "
            "`minus_llr` (positive direction)."
        ),
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

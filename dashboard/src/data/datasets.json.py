"""Observable Framework data loader: dataset-level metadata → JSON.

Per-dataset pins (HF commit, score type, tracking issue, etc.) for the
dataset-metadata block shown above the leaderboard.
"""

from __future__ import annotations

import json
import sys

_METRIC = (
    "AUPRC ± cluster-bootstrap SE — area under the precision-recall curve, "
    "with SE estimated by resampling `match_group`s with replacement "
    "(preserving the matched 1:9 clustering). The `n` in each column "
    "header is the number of match groups."
)

DATASETS = {
    "mendelian_traits": {
        "name": "Mendelian Traits",
        "hf_repo": "bolinas-dna/evals_mendelian_traits",
        "hf_commit": "4aed58e5",
        "score_type": "minus_llr_avg",
        "leading_aggregate": "macro_avg",
        "issue": "https://github.com/Open-Athena/bolinas-dna/issues/161",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "OMIM ∪ HGMD ∪ Smedley et al. 2016 pathogenic SNVs (AF < 0.1%)",
        "negatives": "gnomAD common (AF ≥ 5%)",
        "matching": "1:9 nearest-neighbor on consequence + chrom strata, with continuous TSS/exon distance features (RobustScaler-scaled Euclidean)",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`.",
            "Sorted by Macro Avg by default — the consequence-subset distribution (~92% missense) reflects human-annotator focus on protein-coding disease variants, not the underlying prevalence of pathogenic variants; Global AUPRC therefore over-weights protein-coding-specialist methods. Macro Avg gives equal weight to each subset.",
        ],
    },
    "complex_traits": {
        "name": "Complex Traits",
        "hf_repo": "bolinas-dna/evals_complex_traits",
        "hf_commit": "22f86a89",
        "score_type": "abs_llr_avg",
        "leading_aggregate": "global",
        "issue": "https://github.com/Open-Athena/bolinas-dna/issues/162",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "UKBB fine-mapped complex-trait variants — SuSiE + FINEMAP `max(PIP across traits) > 0.9`",
        "negatives": "`max(PIP) < 0.01` AND no SuSiE/FINEMAP combine-step null-PIP among those traits (`label_variants_by_pip(use_null_pip_guard=True)`)",
        "matching": "1:9 nearest-neighbor on consequence + chrom strata, with continuous TSS/exon distance + MAF features (RobustScaler-scaled Euclidean)",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`. Most consequence subsets in this dataset fall below that threshold — distal and missense are the only ones reported.",
            "Sorted by Global by default. Score column is `abs_llr_avg` (magnitude) rather than `minus_llr_avg` — for complex-trait fine-mapped variants we don't have a pathogenicity direction, only that the variant is causal.",
        ],
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

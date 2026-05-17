"""Observable Framework data loader: dataset-level metadata → JSON.

Per-dataset pins (HF commit, score type, tracking issue, etc.) for the
dataset-metadata block shown above the leaderboard.
"""

from __future__ import annotations

import json
import sys

_METRIC = (
    "PairwiseAccuracy ± Wald SE — fraction of `(positive, negative)` pairs "
    "where the positive scores strictly higher than its matched negative "
    "(ties count 0.5). The `n` in each column header is the number of pairs."
)

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
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`.",
            "Sorted by Macro Avg by default — the consequence-subset distribution (~92% missense) reflects human-annotator focus on protein-coding disease variants, not the underlying prevalence of pathogenic variants; Global PA therefore over-weights protein-coding-specialist methods. Macro Avg gives equal weight to each subset.",
        ],
    },
    "complex_traits": {
        "name": "Complex Traits",
        "hf_repo": "bolinas-dna/evals_complex_traits",
        "hf_commit": "241ac97d",
        "score_type": "abs_llr",
        "leading_aggregate": "global",
        "issue": "https://github.com/Open-Athena/bolinas-dna/issues/162",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "UKBB fine-mapped complex-trait variants — SuSiE + FINEMAP `max(PIP across traits) > 0.9`",
        "negatives": "`max(PIP) < 0.01` AND no SuSiE/FINEMAP combine-step null-PIP among those traits (`label_variants_by_pip(use_null_pip_guard=True)`)",
        "matching": "1:1 on gene, consequence, TSS distance, exon distance, MAF",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`. Most consequence subsets in this dataset fall below that threshold — distal and missense are the only ones reported.",
            "Sorted by Global by default. Score column is `abs_llr` (magnitude) rather than `minus_llr` — for complex-trait fine-mapped variants we don't have a pathogenicity direction, only that the variant is causal.",
        ],
    },
    "eqtl": {
        "name": "eQTL",
        "hf_repo": "bolinas-dna/evals_eqtl",
        "hf_commit": "d32b178d6e",
        "score_type": "abs_llr",
        "leading_aggregate": "global",
        "issue": "https://github.com/Open-Athena/bolinas-dna/issues/172",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "GTEx v8 fine-mapped eQTLs (49 tissues pooled) — SuSiE `max(PIP across tested tissues) > 0.9`",
        "negatives": "`max(PIP) < 0.01` — variant must appear in at least one tissue's nominal sumstats but never reach a strong credible set",
        "matching": "1:1 on gene, consequence, TSS distance, exon distance, MAF (per-subset tiered)",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with `n_pairs < 30`.",
            "Sorted by Global by default. Score column is `abs_llr`; eQTL fine-mapping is direction-agnostic.",
        ],
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

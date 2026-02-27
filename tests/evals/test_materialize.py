"""Tests for sequence materialization into eval harness format."""

from pathlib import Path

import pytest
from biofoundation.data import Genome
from datasets import Dataset

from bolinas.evals.materialize import _add_eval_harness_fields, materialize_sequences

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def mini_genome():
    """Load a small fake genome for deterministic tests.

    Chromosome "1": ACGTACGTACGTACGT... (repeating 4-mer pattern, 50bp)
    Chromosome "2": TTTTTAAAAACCCCCGGGGG + ACGTACGT... (50bp)
    """
    return Genome(FIXTURES_DIR / "mini_genome.fa")


def test_add_eval_harness_fields(mini_genome):
    """Test per-example transform on a known SNV position.

    Chrom "1" sequence: ACGTACGTACGTACGT...
    Variant at pos=11 (1-based) => 0-based index 10 => "G"
    With window_size=10: start=10-5=5, end=10+5=15
    context = genome("1", 5, 10) = "CGTAC"  (positions 5-9)
    right_flank = genome("1", 11, 15) = "TACG"  (positions 11-14)
    ref_completion = "G" + "TACG" = "GTACG"
    alt_completion = "A" + "TACG" = "ATACG"
    """
    example = {"chrom": "1", "pos": 11, "ref": "G", "alt": "A", "label": 1}
    result = _add_eval_harness_fields(example, genome=mini_genome, window_size=10)

    assert result["context"] == "CGTAC"
    assert result["ref_completion"] == "GTACG"
    assert result["alt_completion"] == "ATACG"


def test_window_centering(mini_genome):
    """Verify context and completion lengths sum to window_size."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T", "label": 0}
    result = _add_eval_harness_fields(example, genome=mini_genome, window_size=20)

    assert len(result["context"]) == 10  # window_size // 2
    assert len(result["ref_completion"]) == 10  # window_size // 2
    assert len(result["alt_completion"]) == 10


def test_context_plus_ref_completion_matches_genome(mini_genome):
    """context + ref_completion should reconstruct the reference window."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T", "label": 0}
    result = _add_eval_harness_fields(example, genome=mini_genome, window_size=20)

    full_ref_window = result["context"] + result["ref_completion"]
    # pos=25 (1-based) => center=24 (0-based), start=14, end=34
    expected = mini_genome("1", 14, 34).upper()
    assert full_ref_window == expected


def test_materialize_sequences(mini_genome):
    """End-to-end test: Dataset in, Dataset with new columns out, label renamed."""
    ds = Dataset.from_dict(
        {
            "chrom": ["1", "1"],
            "pos": [11, 15],
            "ref": ["G", "C"],
            "alt": ["A", "T"],
            "label": [1, 0],
        }
    )

    result = materialize_sequences(ds, mini_genome, window_size=10)

    assert "context" in result.column_names
    assert "ref_completion" in result.column_names
    assert "alt_completion" in result.column_names
    assert "target" in result.column_names
    assert "label" not in result.column_names
    assert len(result) == 2


def test_materialize_sequences_preserves_subset(mini_genome):
    """Subset column should be preserved when present."""
    ds = Dataset.from_dict(
        {
            "chrom": ["1", "1"],
            "pos": [11, 15],
            "ref": ["G", "C"],
            "alt": ["A", "T"],
            "label": [1, 0],
            "subset": ["5UTR", "3UTR"],
        }
    )

    result = materialize_sequences(ds, mini_genome, window_size=10)

    assert "subset" in result.column_names
    assert list(result["subset"]) == ["5UTR", "3UTR"]

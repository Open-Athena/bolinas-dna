"""Tests for ``bolinas.pipelines.evals.methods`` — loader + validator for
``dashboard/methods.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bolinas.pipelines.evals.methods import (
    ALL_DATASETS,
    ALL_FAMILIES,
    METHODS_YAML,
    Method,
    load_methods,
    methods_for_dataset,
)


def test_methods_yaml_exists_at_dashboard_path():
    assert METHODS_YAML.exists(), (
        f"methods.yaml missing at {METHODS_YAML}; the loader anchors to the "
        f"repo root and expects dashboard/methods.yaml there"
    )
    assert METHODS_YAML.parent.name == "dashboard"


def test_load_methods_returns_dataclasses():
    methods = load_methods()
    assert len(methods) > 0
    for m in methods:
        assert isinstance(m, Method)
        assert m.family in ALL_FAMILIES
        for d in m.datasets:
            assert d in ALL_DATASETS


def test_load_methods_ids_unique():
    methods = load_methods()
    ids = [m.id for m in methods]
    assert len(ids) == len(set(ids)), (
        f"duplicate id(s) in methods.yaml: {[i for i in ids if ids.count(i) > 1]}"
    )


def test_methods_for_dataset_filters():
    mendelian = methods_for_dataset("mendelian_traits")
    complex_ = methods_for_dataset("complex_traits")
    eqtl = methods_for_dataset("eqtl")
    for m in mendelian:
        assert "mendelian_traits" in m.datasets
    for m in complex_:
        assert "complex_traits" in m.datasets
    for m in eqtl:
        assert "eqtl" in m.datasets


def test_methods_for_dataset_unknown_raises():
    with pytest.raises(AssertionError):
        methods_for_dataset("not_a_dataset")


def test_every_family_has_at_least_one_method():
    methods = load_methods()
    families_present = {m.family for m in methods}
    assert families_present == set(ALL_FAMILIES), (
        f"missing families: {set(ALL_FAMILIES) - families_present}; "
        f"unexpected: {families_present - set(ALL_FAMILIES)}"
    )


def test_bolinas_methods_have_checkpoint(tmp_path: Path):
    methods = load_methods()
    for m in methods:
        if m.family == "bolinas":
            assert m.checkpoint is not None, m.id
            assert m.checkpoint.gcs or m.checkpoint.hf, m.id


def test_invalid_family_rejected(tmp_path: Path):
    bad = tmp_path / "methods.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "x",
                    "display": "x",
                    "family": "not_a_family",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="unknown family"):
        load_methods.__wrapped__(bad)  # type: ignore[attr-defined]


def test_invalid_dataset_rejected(tmp_path: Path):
    bad = tmp_path / "methods.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "x",
                    "display": "x",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["bogus_dataset"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="unknown dataset"):
        load_methods.__wrapped__(bad)  # type: ignore[attr-defined]


def test_duplicate_id_rejected(tmp_path: Path):
    bad = tmp_path / "methods.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "dup",
                    "display": "a",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                },
                {
                    "id": "dup",
                    "display": "b",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                },
            ]
        )
    )
    with pytest.raises(AssertionError, match="duplicate method id"):
        load_methods.__wrapped__(bad)  # type: ignore[attr-defined]


def test_bolinas_requires_checkpoint(tmp_path: Path):
    bad = tmp_path / "methods.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "exp999",
                    "display": "exp999",
                    "family": "bolinas",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="requires `checkpoint:` block"):
        load_methods.__wrapped__(bad)  # type: ignore[attr-defined]

"""Supervised variant-effect prediction on top of gLM embeddings.

Companion library to ``snakemake/analysis/supervised_vep/``. The pipeline caches
mean-pooled embeddings + zero-shot scalars per (model, dataset), and this
module hosts the feature builders, CV splitters, and classifier wrappers used
on top of that cache. See the README of that pipeline for end-to-end usage.
"""

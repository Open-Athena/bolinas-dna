# SkyPilot tasks

Ad-hoc SkyPilot jobs that don't belong in a snakemake pipeline. Docker-based Evo2 inference is the first one.

## `evo2_traitgym.sky.yaml` — Evo2 baseline on TraitGym v2 (issue #131)

Produces per-variant LLR parquets for `evo2_1b_base`, `evo2_7b`, `evo2_40b` on `bolinas-dna/evals-traitgym_mendelian_v2` (train split), at 8192-bp context.

**Always launch with plain `python`.** Evo2's Vortex backend does its own multi-GPU sharding and we let it — no `torchrun`, no HF DDP. For `evo2_1b_base` / `evo2_7b` on an 8-GPU node, Vortex only uses one GPU (the others sit idle); that's the intended behavior. Don't try to data-parallel across ranks, HF Trainer's DDP wrapping fights Vortex's device placement.

Because of that, the 40B model (which requires multiple GPUs) and the smaller ones all share the same 8×H100 cluster. One cluster, three `sky exec` runs.

```bash
# Launch + first run (1B).
sky launch -c evo2 skypilot/evo2_traitgym.sky.yaml \
  --gpus H100:8 \
  --env MODEL=evo2_1b_base

# 7B on the same cluster (skips setup + genome download).
sky exec evo2 skypilot/evo2_traitgym.sky.yaml \
  --env MODEL=evo2_7b

# 40B on the same cluster — Vortex shards pipeline-wise across all 8 GPUs.
sky exec evo2 skypilot/evo2_traitgym.sky.yaml \
  --env MODEL=evo2_40b

# Pull results back to the launching host
rsync -avz sky-evo2:/workspace/results/evo2_traitgym_v2/ results/evo2_traitgym_v2/
```

### Metrics + issue comment

After all three parquets are back locally:

```bash
uv run python scripts/evo2_traitgym_v2_metrics.py
# writes results/evo2_traitgym_v2/metrics.parquet and results_table.md
```

### Tear down

```bash
sky down evo2
```

## Conventions

- Region: `us-east-2` (AWS default per project memory).
- Label: `project=dna` on every instance.
- Reuse clusters across iterations with `sky exec`; only `sky down` at session end.

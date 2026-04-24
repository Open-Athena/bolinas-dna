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

## Insights from running issue #131 baselines (Apr 2026)

### Hardware that worked

| Hardware | Source | Notes |
|---|---|---|
| Lambda `gpu_2x_h100_sxm5` (2×H100 80GB SXM5) | us-south-2 | Full pipeline works. 1B+7B via `torchrun --nproc_per_node=2`. 40B requires Vortex sharding + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. $8.38/hr OD. |
| Lambda `gpu_1x_gh200` (GH200, 96 GB HBM3, ARM64) | us-east-3 | 40B fits on one GPU, no sharding, simpler path. $2.29/hr. **Host is aarch64** — nvcr.io/nvidia/pytorch:25.04-py3 has an arm64 variant (auto-selected by docker). |

### Hardware that failed / was unavailable

- **AWS H100** (p5.4xlarge, p5.48xlarge, p5en.48xlarge, p5e.48xlarge): capacity-out across all zones in every region for ~12 hours. Quotas were fine (L-417A185B = 192 vCPUs OD in us-east-2); issue is pure AWS hardware supply.
- **Lambda `gpu_1x_b200_sxm6`** and `gpu_1x_h100_pcie` / `sxm5`: intermittently capacity-out. Lambda's capacity fluctuates minute-to-minute; `--retry-until-up` works but could take hours.
- **40B on single H100** via data-parallel `torchrun --nproc_per_node=2`: clean OOM on first real-inference batch (model + activations use ~78.5 GB on 80 GB H100 at bs=1, 8192 ctx). Needs ≥96 GB to fit single-GPU.

### Evo2 + biofoundation + HF Trainer pitfalls

1. **`auto_find_batch_size=True` is a no-op for `Trainer.predict()`**. It only wraps the train loop. For inference, tune manually.

2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required for 40B on sharded 2×H100.** Without it, the failure masquerades as a vortex rotary `AttributeError: 'NoneType' object has no attribute 'shape'` (sin_k is None). With it, the same bs=1 run completes cleanly. Classic OOM-in-disguise.

3. **HF Trainer puts inputs on `cuda:0`, but Vortex may shard the embedding onto another device.** Our `compute_evo2_llr` wraps `model.forward` with a small adapter that moves `input_ids` to the embedding's device on entry and moves logits back to the caller's device on exit. No-op on single-GPU.

4. **Running evo2's self-test on a sharded model leaks state** that corrupts the next real inference (hits the same rotary-sin=None error we saw pre-expandable_segments). `SKIP_SELF_TEST=1` is the clean workaround when sharding.

5. **Self-test numerics differ slightly across GPU generations.** Upstream eps is `1e-3`; on GH200 our `evo2_7b_base` test gave loss 0.354 vs expected 0.352 (diff 0.002, fails eps strictly but within rounding).

6. **`sky cancel` does NOT propagate into `sudo docker run` children.** Added a `trap` in the YAML's `run:` block that `sudo docker rm -f $CONTAINER_NAME` on EXIT/INT/TERM so cancellations actually free the GPU. Container is explicitly named via `--name evo2-eval-$MODEL-$$` for this.

7. **`sky launch --retry-until-up` can respawn a cluster even after `sky down`** if the underlying bash process is still alive somewhere. Always verify with `sky status` after `sky down` and reap any ghosts immediately — we lost ~$3 to a ghost `evo2-big` that provisioned in australia-east-1 from a forgotten retry loop.

8. **`sky launch` vs `sky exec`:** `sky launch` on the same cluster rechecks/reruns `setup:`. `sky exec` only runs `run:`. If you've edited setup env vars, `sky launch`. If only the run command changed, `sky exec` (and pass `--gpus` to match the cluster's accelerators, or it will reject with a resource mismatch).

### Throughput summary (8192 ctx, 24,530 TraitGym v2 variants)

| Model | Config | Rate | Full run |
|---|---|---|---|
| 1B | torchrun n=2 bs=8 on 2×H100 | 12.5 v/s | 34m |
| 7B | torchrun n=2 bs=8 on 2×H100 | ~3.3 v/s | 2h 3m |
| 7b_base | python bs=auto on 1×GH200 | ~1.7 v/s | ~4h |
| 40B | python bs=1 on 1×GH200 | 0.35 v/s | ~19h |
| 40B | python bs=2 sharded 2×H100 | 0.23 v/s | ~30h (didn't run) |

### Sanity check

`scripts/evo2_sanity_check.py` validates our `compute_llr_clm` path against `evo2.Evo2.score_sequences(ref) - score_sequences(alt)` (the canonical single-process path from `biofoundation/examples/test_evo2.py`). On the first 10 variants of TraitGym v2 with `evo2_1b_base`, the max |diff| is **0.001 LLR units** — within fp8 noise. Sign convention and sequence construction confirmed correct.

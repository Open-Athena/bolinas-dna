🤖 **Iter 1 — Section A: trace the `win_7_000231918` / `Ceratotherium_simum` example end-to-end**

**Verdict on "is the N-stretch a bug in our region-expansion code?": No.** It's the documented midpoint-centered resize ([`resize.py:1-10`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/src/bolinas/projection/resize.py#L1-L10)) combined with the **absence of a species-side N-filter** (we filter the human anchors against [`undefined.bed`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/windows.smk#L32) but never against any analogous per-species track). Trace below.

## Reproduced from S3 artifacts

| step | command | result |
|---|---|---|
| **A.1** anchor BED lookup | `zcat min0.20.bed.gz \| awk '$4 == "win_7_000231918"'` | `7  29685376  29685631  win_7_000231918` (length 255) |
| **A.2** human anchor in hg38 | `twoBitToFa -seq=7 -start=29685376 -end=29685631 human.2bit -` | **0 N's**, real sequence — human-side filter did its job |
| **A.3** rhino window at projected coords | `twoBitToFa -seq=JH767724.1 -start=24562726 -end=24562981 Ceratotherium_simum.2bit -` | **125 leading N's** + 130 bp of real sequence — matches the HF row |
| **A.6** per-species projection row | `Ceratotherium_simum.parquet` filter on `query_name` | `t_chrom=JH767724.1`, `t_start=24562726`, `t_end=24562981`, `t_strand=+`, `t_src_size=78580661` |

(Slight discrepancy with the OP's "127 N's" — the actual HF-row content has **125** leading N's; off-by-2 is most likely a miscount.)

## The gap on `JH767724.1` near the window

`twoBitInfo -nBed Ceratotherium_simum.2bit` shows exactly **one** N-region overlapping the window:

```
JH767724.1   24,562,271   24,562,851   (length 580 bp)
```

Nearby N-regions on the same scaffold (within ±100 kb) — these are isolated, no compounding:

```
JH767724.1   24,512,952   24,513,337   (385 bp, ~50 kb upstream)
JH767724.1   24,557,344   24,557,444   (100 bp, ~5 kb upstream)
JH767724.1   24,562,271   24,562,851   (580 bp, the offender)
JH767724.1   24,674,332   24,674,432   (100 bp, ~111 kb downstream)
JH767724.1   24,696,391   24,696,807   (416 bp, ~133 kb downstream)
```

## Window position within the gap

```
              ←————— 580 bp gap (24,562,271 → 24,562,851) —————→
              ↓                                                ↓
              ............................................NNNN........real sequence...
                                          ↑                       ↑
                       24,562,726  ←——————————— 255 bp window ———————————→  24,562,981
                              first 125 bp inside the gap = N      last 130 bp outside = real
```

So the window's 5′ end sits **125 bp inside** the gap and the 3′ end sits **130 bp past** it. The 125 leading N's match exactly: window start (24,562,726) → gap end (24,562,851) = 125 bp of overlap.

## Mechanism attribution

This pattern is **consistent with M2 (resize expansion into a gap)**:

- The lifted region's midpoint maps near `(24,562,726 + 24,562,981) / 2 = 24,562,853` — just **2 bp past** the gap's 3′ edge. That's where `halLiftover` likely placed the orthologous center.
- Midpoint-centered resize to 255 bp on a (probably) short lift (length ∈ [128, 512] per [`filter_length`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/src/bolinas/projection/filter.py#L53-L59)) then expanded ~127 bp on each side. The leftward expansion crossed straight into the gap.

Distinguishing M2 from M1 (`halLiftover` itself returning a span inside the gap) **exactly** would require re-running `halLiftover` against the HAL for this anchor — the per-species parquet only persists post-resize coords. Section B's per-window N-position decomposition (5′-edge vs 3′-edge vs interior) on the full dataset will tell us which mechanism dominates without needing per-anchor halLiftover.

## What this means for the dataset

- Not a code bug. Resize behaves as documented (`resize.py` explicitly notes "padded bases beyond the orthologous extent are not orthologous… for a fixed-context gLM that is acceptable").
- The 580 bp gap is **wider than our 255 bp window**, so any anchor whose midpoint maps within ±127 bp of either gap edge will produce a partially-N row. No species-side filter ever catches that.
- The cleanest fix candidate so far is **F1 (post-extraction N-filter)** — a single `n_frac` column + threshold in [`dataset.smk`](https://github.com/Open-Athena/bolinas-dna/blob/a38154132b173bffe0fb5d589682656b327759fd/snakemake/zoonomia_projection_dataset/workflow/rules/dataset.smk). Mechanism-agnostic, minimal code change. Section B will quantify the fraction of rows this affects.

Next iteration: Section B (dataset-wide N quantification, joined with assembly metadata).

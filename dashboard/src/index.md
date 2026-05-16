---
title: Mendelian Traits Leaderboard
toc: false
---

# Mendelian Traits Leaderboard

Matched-pair PairwiseAccuracy ± SE per method on the [`bolinas-dna/evals_mendelian_traits`](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits) dataset (ClinVar pathogenic vs. gnomAD-common-frequency, 1:1 gene-matched). Higher is better; random baseline is 0.5.

This is Phase B — a plain interactive table. [Phase C](https://github.com/Open-Athena/bolinas-dna/blob/main/.claude/plans/see-leaderboards-at-https-github-com-ope-zazzy-quail.md) will swap it for a color-encoded heatmap pinned at the [0.5, 1.0] range, with per-family filters and tooltips.

```js
const leaderboard = await FileAttachment("data/leaderboard.parquet").parquet();
const methods = await FileAttachment("data/methods.json").json();
const datasets = await FileAttachment("data/datasets.json").json();
```

```js
// Pull rows + index by id for the table.
const allRows = leaderboard.toArray().map(r => ({
  method_id: String(r.method_id),
  method_display: String(r.method_display),
  family: String(r.family),
  subset: String(r.subset),
  value: Number(r.value),
  se: Number(r.se),
  n_pairs: Number(r.n_pairs),
  n_ties: Number(r.n_ties),
  dataset: String(r.dataset),
}));
const mendelian = allRows.filter(r => r.dataset === "mendelian_traits");
const methodById = new Map(methods.map(m => [m.id, m]));
```

## Dataset

```js
const meta = datasets.mendelian_traits;
const hfUrl = `https://huggingface.co/datasets/${meta.hf_repo}/tree/${meta.hf_commit}`;
```

```js
display(html`<div class="card">
  <table>
    <tr><td><b>HF dataset</b></td><td><a href=${hfUrl}><code>${meta.hf_repo} @ ${meta.hf_commit}</code></a></td></tr>
    <tr><td><b>Split</b></td><td><code>${meta.split}</code> (test held out for final-eval)</td></tr>
    <tr><td><b>Score column</b></td><td><code>${meta.score_type}</code> (higher should be more pathogenic)</td></tr>
    <tr><td><b>Subset threshold</b></td><td><code>n_pairs ≥ ${meta.n_min_per_subset}</code></td></tr>
    <tr><td><b>Sort axis</b></td><td>Macro Avg (rationale: ~92% missense over-weights protein-coding-specialist methods on Global)</td></tr>
    <tr><td><b>Tracking issue</b></td><td><a href=${meta.issue}>${meta.issue.replace("https://github.com/", "")}</a></td></tr>
  </table>
  <p>${meta.description}</p>
</div>`);
```

## Macro Avg ranking

PairwiseAccuracy averaged unweighted across per-consequence subsets with `n_pairs ≥ 30`. The leftmost row is the current leader on this aggregate.

```js
const macroRows = mendelian
  .filter(r => r.subset === "_macro_avg_")
  .sort((a, b) => b.value - a.value)
  .map(r => ({
    method: r.method_display,
    family: r.family,
    "Macro Avg": r.value,
    SE: r.se,
    "Subsets ≥ 30": r.n_pairs,
    description: methodById.get(r.method_id)?.description ?? "",
  }));
```

```js
Inputs.table(macroRows, {
  rows: 30,
  format: {
    "Macro Avg": x => x.toFixed(3),
    SE: x => x.toFixed(3),
  },
  width: {
    method: 220,
    family: 110,
    "Macro Avg": 90,
    SE: 70,
    "Subsets ≥ 30": 110,
  },
})
```

## Global ranking

PairwiseAccuracy across **all** matched pairs, regardless of per-subset size. Same data, alternate aggregate.

```js
const globalRows = mendelian
  .filter(r => r.subset === "_global_")
  .sort((a, b) => b.value - a.value)
  .map(r => ({
    method: r.method_display,
    family: r.family,
    Global: r.value,
    SE: r.se,
    "n pairs": r.n_pairs,
    description: methodById.get(r.method_id)?.description ?? "",
  }));
```

```js
Inputs.table(globalRows, {
  rows: 30,
  format: {
    Global: x => x.toFixed(3),
    SE: x => x.toFixed(3),
  },
  width: {
    method: 220,
    family: 110,
    Global: 90,
    SE: 70,
    "n pairs": 90,
  },
})
```

## Per-subset breakdown

Long-form: one row per (method, subset). Subsets with `n_pairs < 30` are kept here but flagged. Filter the table directly with the column controls.

```js
const subsetRows = mendelian
  .filter(r => r.subset !== "_global_" && r.subset !== "_macro_avg_")
  .sort((a, b) => b.value - a.value)
  .map(r => ({
    method: r.method_display,
    family: r.family,
    subset: r.subset,
    PA: r.value,
    SE: r.se,
    "n pairs": r.n_pairs,
    flagged: r.n_pairs < meta.n_min_per_subset ? "yes" : "",
  }));
```

```js
Inputs.table(subsetRows, {
  rows: 40,
  format: {
    PA: x => x.toFixed(3),
    SE: x => x.toFixed(3),
  },
  width: {
    method: 220,
    family: 110,
    subset: 180,
    PA: 70,
    SE: 70,
    "n pairs": 80,
    flagged: 80,
  },
})
```

---

<small>Phase B build of the dashboard described in <a href="https://github.com/Open-Athena/bolinas-dna/blob/main/.claude/plans/see-leaderboards-at-https-github-com-ope-zazzy-quail.md">the plan</a>. The heatmap UI (Phase C) replaces this plain-table view once the layout is validated.</small>

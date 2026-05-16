---
title: Mendelian Traits Leaderboard
toc: false
---

# Mendelian Traits Leaderboard

Matched-pair PairwiseAccuracy per method on the [`bolinas-dna/evals_mendelian_traits`](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits) dataset (ClinVar pathogenic vs. gnomAD-common-frequency, 1:1 gene-matched within consequence subset). Higher is better; random baseline is 0.5.

```js
const leaderboard = await FileAttachment("data/leaderboard.parquet").parquet();
const methods = await FileAttachment("data/methods.json").json();
const datasets = await FileAttachment("data/datasets.json").json();
import {heatmap, colorLegend} from "./components/heatmap.js";
```

```js
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
const meta = datasets.mendelian_traits;
```

## Dataset

```js
const hfUrl = `https://huggingface.co/datasets/${meta.hf_repo}/tree/${meta.hf_commit}`;
```

```js
display(html`<div class="card">
  <table class="dataset-meta">
    <tr><td><b>HF dataset</b></td><td><a href=${hfUrl}><code>${meta.hf_repo} @ ${meta.hf_commit}</code></a></td></tr>
    <tr><td><b>Split</b></td><td><code>${meta.split}</code> (test held out for final-eval)</td></tr>
    <tr><td><b>Score column</b></td><td><code>${meta.score_type}</code> (higher should be more pathogenic)</td></tr>
    <tr><td><b>Subset threshold</b></td><td><code>n_pairs ≥ ${meta.n_min_per_subset}</code></td></tr>
    <tr><td><b>Sort axis</b></td><td>Macro Avg by default (rationale: ~92% missense over-weights protein-coding-specialist methods on Global). Click any column header to re-sort.</td></tr>
    <tr><td><b>Tracking issue</b></td><td><a href=${meta.issue}>${meta.issue.replace("https://github.com/", "")}</a></td></tr>
  </table>
  <p>${meta.description}</p>
</div>`);
```

## Leaderboard

```js
const families = ["bolinas", "conservation", "alphagenome", "gpn_star"];
const familyChoice = view(
  Inputs.checkbox(families, {
    label: "Family",
    value: families,
  }),
);
const search = view(
  Inputs.search({
    label: "Method name",
    placeholder: "filter by method (e.g. exp166, phyloP)",
  }),
);
```

```js
const filtered = mendelian.filter(r => {
  if (!familyChoice.includes(r.family)) return false;
  if (search && !r.method_display.toLowerCase().includes(search.toLowerCase())) return false;
  return true;
});
```

<style>
.lb-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 1em 0;
}
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 8px;
  border: 1px solid #ddd;
}
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  cursor: pointer;
  user-select: none;
  min-width: 70px;
}
.lb-heatmap thead th:hover { background: #eee; }
.lb-col-sorted { background: #e8f0e8 !important; font-weight: 600; }
.lb-method-header { text-align: left !important; cursor: default !important; }
.lb-method { white-space: nowrap; }
.lb-method a { text-decoration: none; }
.lb-method a:hover { text-decoration: underline; }
.lb-desc { color: #666; font-size: 0.92em; }
.lb-family {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}
.lb-family-bolinas      { background: #1f77b4; }
.lb-family-conservation { background: #7f7f7f; }
.lb-family-alphagenome  { background: #d62728; }
.lb-family-gpn_star     { background: #9467bd; }
.lb-cell {
  text-align: center;
  font-feature-settings: "tnum";
}
.lb-na { text-align: center; color: #aaa; }
.dataset-meta td { padding: 2px 8px; }
.dataset-meta td:first-child { white-space: nowrap; }
.legend-row {
  display: flex; align-items: center; gap: 12px;
  margin: 0.5em 0 1em;
  font-size: 0.85em; color: #444;
}
</style>

<div class="legend-row">
  <span>Color: PA, fixed scale</span>
  ${colorLegend({width: 240, height: 14})}
  <span>· Hover a cell for SE + n_pairs · Click a column header to re-sort</span>
</div>

```js
display(
  heatmap({
    rows: filtered,
    methodById,
    leadingAggregate: meta.leading_aggregate === "macro_avg" ? "_macro_avg_" : "_global_",
  }),
);
```

## Notes

- **Bolding has been dropped**: cell colors carry the same ranking signal more cleanly than `**bold**` did in the old issue tables. Hover any cell for `PA ± SE`, `n_pairs`, and `n_ties`.
- **Below random (< 0.5)** clamps to white. A method underperforming random on a subset reads as "no signal" rather than as a separate red regime.
- **Missing cells (—)** mean the method wasn't evaluated on that subset (e.g. mendelian-only entries on the `5' UTR` column for a conservation-track-only run, where applicable).

For the raw data, see `/data/leaderboard.parquet` (long-form) and `/data/methods.json`; the [About page](./about) documents the full schema.

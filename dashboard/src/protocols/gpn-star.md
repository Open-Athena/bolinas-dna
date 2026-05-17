---
title: GPN-Star protocol comparison
toc: false
wide: true
---

# GPN-Star: protocol comparison

```js
const leaderboard = await FileAttachment("../data/leaderboard.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
import {heatmap, colorLegend} from "../components/heatmap.js";
```

```js
const FAMILY = "gpn_star";
const PROTOCOLS = ["cLLR", "LLR"];

const allRows = leaderboard.toArray().map(r => ({
  method_id: String(r.method_id),
  method_display: String(r.method_display),
  family: String(r.family),
  protocol: String(r.protocol),
  subset: String(r.subset),
  value: Number(r.value),
  se: Number(r.se),
  n_pairs: Number(r.n_pairs),
  n_ties: Number(r.n_ties),
  dataset: String(r.dataset),
}));

const grouped = new Map();
for (const r of allRows) {
  if (r.family !== FAMILY || r.dataset !== "mendelian_traits") continue;
  const key = r.method_id + "\0" + r.subset;
  if (!grouped.has(key)) grouped.set(key, {method_id: r.method_id, method_display: r.method_display, subset: r.subset});
  grouped.get(key)[r.protocol] = r;
}

const modelById = new Map(methods.map(m => [m.id, m]));
```

```js
const baseline = view(
  Inputs.select(PROTOCOLS, {label: "Baseline", value: "cLLR"}),
);
const alternative = view(
  Inputs.select(PROTOCOLS, {label: "Compared with", value: "LLR"}),
);
```

```js
const deltaRows = [];
if (baseline !== alternative) {
  for (const cell of grouped.values()) {
    const d = cell[baseline];
    const a = cell[alternative];
    if (!d || !a) continue;
    deltaRows.push({
      method_id: cell.method_id,
      method_display: cell.method_display,
      family: FAMILY,
      protocol: alternative,
      subset: cell.subset,
      value: a.value - d.value,
      se: 0,
      n_pairs: d.n_pairs,
      n_ties: 0,
      dataset: "mendelian_traits",
    });
  }
}
```

```js
const sortKeyState = Mutable("_macro_avg_");
const setSortKey = (k) => { sortKeyState.value = k; };
```

Each cell below is **${alternative} PA − ${baseline} PA**, in percentage points. Green = ${alternative} scores higher than ${baseline}; red = the reverse. cLLR is the producer's recommended protocol on this leaderboard ([Benegas et al. #145](https://github.com/Open-Athena/bolinas-dna/issues/145)) — calibration subtracts pentanucleotide-context background, `llr_calibrated = llr − E[llr | 5-mer, mut]`.

Same matched pairs as the [Mendelian leaderboard](../leaderboards/mendelian); only the `score_type` filter changes.

<style>
:root { --observablehq-max-width: 1920px; }
main > p, main > h1, main > h2, main > h3, main > .card { max-width: none; }
main > p, main > h1, main > h2, main > h3 { max-width: 1100px; }

.lb-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 1em 0;
  table-layout: fixed;
  width: 1170px;
}
.lb-heatmap thead tr { height: 40px; }
.lb-heatmap tbody tr { height: 28px; }
.lb-heatmap-row { display: block; }
.lb-heatmap thead th:not(.lb-method-header) { width: 96px; }
.lb-heatmap th.lb-method-header { width: 210px; }
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 4px;
  border: 1px solid #ddd;
}
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  cursor: pointer;
  user-select: none;
}
.lb-heatmap thead th:hover { background: #eee; }
.lb-col-label { white-space: nowrap; }
.lb-heatmap thead th.lb-col-sorted { background: #d6e8d6; font-weight: 600; }
.lb-heatmap td.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.lb-heatmap thead th.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; border-bottom: 2px solid #5c8a5c; }
.lb-method {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lb-method a {
  text-decoration: none;
  display: inline-block;
  max-width: calc(100% - 22px);
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
.lb-method a:hover { text-decoration: underline; }
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
.lb-cell { text-align: center; font-feature-settings: "tnum"; }
.lb-na { text-align: center; color: #aaa; }
</style>

```js
display(
  heatmap({
    rows: deltaRows,
    modelById,
    sortKey: sortKeyState,
    onSortChange: setSortKey,
    leadingAggregate: "_macro_avg_",
    palette: "delta",
    showForest: false,
  }),
);
```

<small>Color scale clamps at ±10 percentage points. Sort by clicking any column header — sort survives navigation. The model column links to the model card on the [Models](/models) page.</small>

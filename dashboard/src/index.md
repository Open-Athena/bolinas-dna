---
title: Mendelian Traits Leaderboard
toc: false
wide: true
---

# Mendelian Traits Leaderboard

Matched-pair PairwiseAccuracy per method on the [`bolinas-dna/evals_mendelian_traits`](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits) dataset (HGMD ∪ OMIM ∪ Smedley 2016 pathogenic SNVs vs. gnomAD common-frequency, 1:1 gene-matched within consequence subset). Higher is better; random baseline is 0.5.

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
  </table>
  <p>${meta.description}</p>
</div>`);
```

## Leaderboard

```js
const families = ["bolinas", "conservation", "alphagenome", "gpn_star"];
```

```js
function FamilyToggle(allFamilies, initial = allFamilies) {
  const state = new Set(initial);
  const node = html`<div class="lb-family-toggle"></div>`;
  let _value = [...state];
  Object.defineProperty(node, "value", {get: () => _value});
  function fire() {
    _value = [...state];
    node.dispatchEvent(new Event("input", {bubbles: true}));
  }
  function setAll(next) {
    state.clear();
    next.forEach((f) => state.add(f));
    render();
    fire();
  }
  function render() {
    node.replaceChildren(html`<div class="lb-family-toggle-row">
      ${allFamilies.map(
        (f) => html`<button
          type="button"
          class=${`lb-pill family-${f}${state.has(f) ? " active" : ""}`}
          aria-pressed=${state.has(f) ? "true" : "false"}
          onclick=${() => {
            state.has(f) ? state.delete(f) : state.add(f);
            render();
            fire();
          }}
        >${f}</button>`,
      )}
      <span class="lb-toggle-actions">
        <button type="button" class="lb-link" onclick=${() => setAll(allFamilies)}>all</button>
        <span aria-hidden="true">·</span>
        <button type="button" class="lb-link" onclick=${() => setAll([])}>none</button>
      </span>
    </div>`);
  }
  render();
  return node;
}

const familyChoice = view(FamilyToggle(families));
const search = view(
  Inputs.text({
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
/* Observable Framework's default theme caps prose elements at 640px.
   Override on this page so the intro paragraph + dataset card align
   with the wider leaderboard table below. */
main > p, main > h1, main > h2, main > h3, main > .card {
  max-width: none;
}
main > h1, main > h2, main > h3, main > p { max-width: 1200px; }
.card { max-width: 1200px; }

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

/* Family pill toggle */
.lb-family-toggle-row {
  display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
  margin: 0.25em 0 0.75em;
}
.lb-pill {
  appearance: none;
  border: 1.5px solid transparent;
  background: transparent;
  border-radius: 9999px;
  padding: 3px 10px;
  font: inherit;
  font-size: 0.85em;
  cursor: pointer;
  transition: background 80ms, border-color 80ms, color 80ms;
  color: #999;
}
.lb-pill:not(.active) {
  border-color: #ddd;
}
.lb-pill:not(.active):hover {
  border-color: #999;
  color: #555;
}
.lb-pill.active {
  color: #fff;
  border-color: transparent;
}
.lb-pill.active.family-bolinas      { background: #1f77b4; }
.lb-pill.active.family-conservation { background: #7f7f7f; }
.lb-pill.active.family-alphagenome  { background: #d62728; }
.lb-pill.active.family-gpn_star     { background: #9467bd; }
.lb-toggle-actions {
  margin-left: 6px;
  color: #888;
  font-size: 0.82em;
}
.lb-toggle-actions .lb-link {
  appearance: none;
  background: transparent;
  border: none;
  padding: 0 2px;
  font: inherit;
  font-size: inherit;
  color: #3a7bd5;
  cursor: pointer;
}
.lb-toggle-actions .lb-link:hover { text-decoration: underline; }
</style>

<div class="legend-row">
  <span>Color: PA, fixed scale</span>
  ${colorLegend({width: 240, height: 14})}
  <span>· Click a column header to re-sort</span>
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

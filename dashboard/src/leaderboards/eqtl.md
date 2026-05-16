---
title: eQTL Leaderboard
toc: false
wide: true
---

# eQTL Leaderboard

```js
const leaderboard = await FileAttachment("../data/leaderboard.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {heatmap, colorLegend} from "../components/heatmap.js";
```

```js
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
const eqtl = allRows.filter(r => r.dataset === "eqtl");
const modelById = new Map(methods.map(m => [m.id, m]));
const meta = datasets.eqtl;
```

```js
// Sort column state — lives outside the heatmap so it survives
// re-mounts when family / protocol / search filters change. This cell
// runs once per page load (when `meta` resolves); the Mutable persists
// across all later heatmap re-renders.
const sortKeyState = Mutable(
  meta.leading_aggregate === "macro_avg" ? "_macro_avg_" : "_global_",
);
const setSortKey = (k) => {
  sortKeyState.value = k;
};
```

## Dataset

```js
const hfUrl = `https://huggingface.co/datasets/${meta.hf_repo}/tree/${meta.hf_commit}`;
```

```js
display(html`<div class="card">
  <table class="dataset-meta">
    <tr><td><b>HF dataset</b></td><td><a href=${hfUrl}><code>${meta.hf_repo} @ ${meta.hf_commit}</code></a></td></tr>
    <tr><td><b>Split</b></td><td><code>${meta.split}</code> (used for both training and development; test held out for final-eval)</td></tr>
  </table>
  <div class="dataset-bullets">
    <div><b>Positives:</b> ${meta.positives}</div>
    <div><b>Negatives:</b> ${meta.negatives}</div>
    <div><b>Matching:</b> ${meta.matching}</div>
    <div><b>Metric:</b> ${meta.metric}</div>
  </div>
  <div class="dataset-notes">
    ${meta.notes.map((n) => html`<div>${n}</div>`)}
  </div>
</div>`);
```

## Leaderboard

```js
const families = ["bolinas", "conservation", "alphagenome", "gpn_star"];
// Display labels for families (used in pill toggle, protocol picker,
// popover chip). The slug stays the data-layer key (PROTOCOLS, parquet
// rows, CSS classes); only the rendered text is humanised.
const FAMILY_LABEL = {
  bolinas: "bolinas",
  conservation: "conservation",
  alphagenome: "AlphaGenome",
  gpn_star: "GPN-Star",
};
// Protocol options per family. Mirror of `PROTOCOLS` in
// src/bolinas/pipelines/evals/leaderboard.py — keep in sync when adding
// a protocol. Defaults match `DEFAULT_PROTOCOL`.
const PROTOCOL_OPTIONS = {
  bolinas: ["LLR", "JSD"],
  gpn_star: ["cLLR", "LLR"],
};
const PROTOCOL_DEFAULTS = {
  bolinas: "LLR",
  conservation: "score",
  alphagenome: "L2",
  gpn_star: "cLLR",
};
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
        >${FAMILY_LABEL[f] ?? f}</button>`,
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
```

```js
function ProtocolPicker(options, defaults) {
  // options: {family: [protocol_name, ...]}; defaults: {family: protocol_name}.
  // Only families with multiple options get rendered as a toggle.
  const state = {...defaults};
  const node = html`<div class="lb-protocol-picker"></div>`;
  Object.defineProperty(node, "value", {get: () => ({...state})});
  function fire() {
    node.dispatchEvent(new Event("input", {bubbles: true}));
  }
  function render() {
    const groups = [];
    for (const [fam, protos] of Object.entries(options)) {
      if (protos.length < 2) continue;
      groups.push(html`<span class="lb-protocol-group">
        <span class=${`lb-family-tag family-${fam}`}>${FAMILY_LABEL[fam] ?? fam}</span>
        <span class="lb-protocol-segmented">${protos.map(p => html`<button
          type="button"
          class=${`lb-protocol-btn${state[fam] === p ? " active" : ""}`}
          onclick=${() => { state[fam] = p; render(); fire(); }}
        >${p}</button>`)}</span>
      </span>`);
    }
    node.replaceChildren(html`<div class="lb-protocol-picker-row">${groups}</div>`);
  }
  render();
  return node;
}

const protocolChoice = view(ProtocolPicker(PROTOCOL_OPTIONS, PROTOCOL_DEFAULTS));
const search = view(
  Inputs.text({
    label: "Model name",
    placeholder: "filter by method (e.g. exp166, phyloP)",
  }),
);
```

```js
const filtered = eqtl.filter(r => {
  if (!familyChoice.includes(r.family)) return false;
  // One protocol per family. Falls back to DEFAULTS for families that
  // don't appear in `protocolChoice` (single-option families).
  const wantedProtocol = protocolChoice[r.family] ?? PROTOCOL_DEFAULTS[r.family];
  if (r.protocol !== wantedProtocol) return false;
  if (search && !r.method_display.toLowerCase().includes(search.toLowerCase())) return false;
  return true;
});
```

<style>
/* Observable Framework's default theme caps prose elements at 640px and
   constrains main to ~1072px even on a wide page. Override both so the
   heatmap and the side-by-side forest plot fit on one row. */
:root { --observablehq-max-width: 2200px; }
main > p, main > h1, main > h2, main > h3, main > .card {
  max-width: none;
}
main > h1, main > h2, main > h3, main > p { max-width: 1200px; }
.card { max-width: 1200px; }
.lb-heatmap-row { width: max-content; max-width: 100%; }
.lb-heatmap, .lb-forest { flex: 0 0 auto; }

.lb-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 1em 0;
  /* Fixed layout with an explicit overall width so every data column
     shares one identical 90px width. 90px fits the widest header
     ("Synonymous") with padding to spare; the model column takes the
     remainder. */
  table-layout: fixed;
  width: 1170px;
}
.lb-heatmap thead th:not(.lb-method-header) { width: 90px; }
.lb-heatmap th.lb-method-header { width: 210px; }
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 4px;
  border: 1px solid #ddd;
}
/* Explicit row heights so the forest plot to the right (which uses fixed
   pixel rowH) lines up dot-for-row with the heatmap. Keep in sync with
   HEATMAP_HEADER_PX / HEATMAP_ROW_PX in dashboard/src/components/heatmap.js. */
.lb-heatmap thead tr { height: 40px; }
.lb-heatmap tbody tr { height: 28px; }
.lb-heatmap-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  flex-wrap: nowrap;
  overflow-x: auto;
}
/* Drop the table's outer top margin inside the side-by-side row so the
   heatmap's top edge lines up with the forest plot's. */
.lb-heatmap-row .lb-heatmap { margin: 0; }
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  cursor: pointer;
  user-select: none;
}
.lb-heatmap td.lb-cell { font-variant-numeric: tabular-nums; padding: 6px 4px; }
.lb-heatmap thead th:hover { background: #eee; }
.lb-col-label { white-space: nowrap; }
.lb-heatmap thead th.lb-col-sorted { background: #d6e8d6; font-weight: 600; }
/* Bold left/right borders + slight inset shadow mark the sorted column
   on the body rows (not just the header). */
.lb-heatmap td.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.lb-heatmap thead th.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; border-bottom: 2px solid #5c8a5c; }
.lb-method-header { text-align: left !important; cursor: default !important; }
.lb-method {
  /* Long model names truncate with an ellipsis. Hover popover surfaces
     the full name + family + links, so info is one hover away. */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lb-method a {
  text-decoration: none;
  display: inline-block;
  max-width: calc(100% - 22px); /* room for the family swatch + margin */
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
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
.lb-forest { display: block; }
.lb-forest-empty { color: #888; margin: 1em 0; font-size: 0.9em; }
.dataset-meta td { padding: 2px 8px; }
.dataset-meta td:first-child { white-space: nowrap; }
.dataset-bullets { margin: 0.5em 0 0.25em; }
.dataset-bullets div { margin: 2px 0; }
.dataset-notes { margin-top: 0.5em; color: #666; font-size: 0.9em; }
.dataset-notes div { margin: 2px 0; }
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

/* Per-family protocol picker (segmented toggle row) */
.lb-protocol-picker-row {
  display: flex; align-items: center; flex-wrap: wrap; gap: 16px;
  margin: 0.25em 0 0.75em;
  font-size: 0.85em;
}
.lb-protocol-group {
  display: inline-flex; align-items: center; gap: 6px;
}
.lb-family-tag {
  display: inline-block;
  font-family: var(--monospace);
  font-size: 0.85em;
  padding: 1px 7px;
  border-radius: 9999px;
  color: #fff;
}
.lb-family-tag.family-bolinas      { background: #1f77b4; }
.lb-family-tag.family-conservation { background: #7f7f7f; }
.lb-family-tag.family-alphagenome  { background: #d62728; }
.lb-family-tag.family-gpn_star     { background: #9467bd; }
.lb-protocol-segmented {
  display: inline-flex;
  border: 1px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
}
.lb-protocol-btn {
  appearance: none;
  background: #fff;
  border: none;
  border-left: 1px solid #ccc;
  padding: 2px 9px;
  font: inherit;
  font-size: 0.95em;
  color: #555;
  cursor: pointer;
  transition: background 80ms, color 80ms;
}
.lb-protocol-btn:first-child { border-left: none; }
.lb-protocol-btn:hover:not(.active) { background: #f4f4f4; color: #000; }
.lb-protocol-btn.active {
  background: #333;
  color: #fff;
}

/* Model popover on heatmap method-name hover */
.lb-method-popover {
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
  font-size: 0.85em;
  line-height: 1.4;
  padding: 10px 12px;
  min-width: 280px;
  max-width: 360px;
}
.lb-pop-header { display: flex; flex-direction: column; gap: 3px; margin-bottom: 4px; }
.lb-pop-family {
  display: inline-block;
  font-size: 0.7em;
  padding: 1px 7px;
  border-radius: 9999px;
  color: #fff;
  width: fit-content;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.lb-pop-family.family-bolinas      { background: #1f77b4; }
.lb-pop-family.family-conservation { background: #7f7f7f; }
.lb-pop-family.family-alphagenome  { background: #d62728; }
.lb-pop-family.family-gpn_star     { background: #9467bd; }
.lb-pop-display { font-size: 0.98em; font-weight: 600; }
.lb-pop-desc { color: #555; margin: 4px 0 6px; font-size: 0.92em; }
.lb-pop-specs { margin: 6px 0; }
.lb-pop-row {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 10px;
  align-items: baseline;
  margin: 2px 0;
}
.lb-pop-key {
  color: #888; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em;
}
.lb-pop-val { font-size: 0.92em; }
.lb-pop-links { margin: 6px 0 2px; font-size: 0.9em; }
.lb-pop-more {
  display: inline-block;
  margin-top: 6px;
  font-size: 0.82em;
  color: #3a7bd5;
}
.muted { color: #aaa; }
</style>

<div class="legend-row">
  <span>Color: PA, fixed scale</span>
  ${colorLegend({width: 240, height: 14})}
  <span>· Click a column header to re-sort</span>
</div>

```js
// Reactive on `filtered` (family/protocol/search) AND on `sortKeyState`
// (column click). Observable Framework auto-unwraps a Mutable when read
// from another cell — `sortKeyState` here is the current string value,
// not the Mutable instance.
display(
  heatmap({
    rows: filtered,
    modelById,
    sortKey: sortKeyState,
    onSortChange: setSortKey,
    leadingAggregate: meta.leading_aggregate === "macro_avg" ? "_macro_avg_" : "_global_",
  }),
);
```

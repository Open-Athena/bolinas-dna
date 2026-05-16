// Leaderboard heatmap: method × subset → PA color cell.
//
// Sequential YlGn color scale fixed at [0.5, 1.0]:
//   - 0.5 (random baseline) = pale yellow (neutral)
//   - 1.0 (perfect)        = deep green
// PA values below 0.5 are clamped to 0.5 — they're treated as noise rather
// than highlighted as anti-predictive. The scale stays pinned across filter
// selections so cells are comparable.

import * as d3 from "npm:d3";
import {html, svg} from "npm:htl";

import {attachModelPopover} from "./model-cards.js";

const SUBSET_DISPLAY = {
  missense_variant: "Missense",
  splicing: "Splicing",
  "5_prime_UTR_variant": "5' UTR",
  distal: "Distal",
  "3_prime_UTR_variant": "3' UTR",
  tss_proximal: "Promoter",
  non_coding_transcript_exon_variant: "ncRNA",
  synonymous_variant: "Synonymous",
};

const GLOBAL = "_global_";
const MACRO = "_macro_avg_";
const N_MIN = 30;

// Sequential YlGn on [0.5, 1.0]. Below-random PA clamps to 0.5 so it just
// reads as "no signal" instead of as a separate regime. Skip the lightest
// 10% of the interpolator so 0.5 still has a faint yellow wash, making it
// distinguishable from "missing cell".
function paColor(v) {
  if (v == null) return "#ffffff";
  const clamped = Math.max(0.5, Math.min(1.0, v));
  return d3.interpolateYlGn(0.1 + 0.85 * ((clamped - 0.5) / 0.5));
}

// Text contrast against the cell background by Lab lightness.
function textColor(v) {
  if (v == null) return "#666";
  return d3.lab(paColor(v)).l > 60 ? "#000" : "#fff";
}

function fmt(v, se) {
  return `${v.toFixed(3)} ± ${se.toFixed(3)}`;
}

// Sort: numeric desc, with method_display as a stable tiebreaker so toggle
// behavior is predictable.
function makeComparator(sortKey) {
  return (a, b) => {
    const va = a.cells.get(sortKey)?.value ?? -Infinity;
    const vb = b.cells.get(sortKey)?.value ?? -Infinity;
    if (va !== vb) return vb - va;
    return a.method_display.localeCompare(b.method_display);
  };
}

/**
 * Build the heatmap.
 *
 * @param {object} opts
 * @param {Array} opts.rows long-form (method × subset) rows; output of
 *   `bolinas.pipelines.evals.leaderboard.normalized_rows`.
 * @param {Map}   opts.modelById metadata for each method id (for links + tooltips).
 * @param {string} opts.leadingAggregate "_macro_avg_" or "_global_" — drives the
 *   default sort + which aggregate appears leftmost.
 * @returns {HTMLElement}
 */
export function heatmap({rows, modelById, leadingAggregate = MACRO}) {
  // Group by method_id; collect cells in a Map keyed by subset.
  const byMethod = new Map();
  for (const r of rows) {
    if (!byMethod.has(r.method_id)) {
      byMethod.set(r.method_id, {
        method_id: r.method_id,
        method_display: r.method_display,
        family: r.family,
        cells: new Map(),
      });
    }
    byMethod
      .get(r.method_id)
      .cells.set(r.subset, {value: r.value, se: r.se, n_pairs: r.n_pairs, n_ties: r.n_ties});
  }

  // Per-subset max n_pairs across methods (for the "(n=X)" header label
  // and the n>=30 filter).
  const subsetMaxN = new Map();
  for (const m of byMethod.values()) {
    for (const [subset, cell] of m.cells) {
      if (subset === GLOBAL || subset === MACRO) continue;
      subsetMaxN.set(subset, Math.max(subsetMaxN.get(subset) ?? 0, cell.n_pairs));
    }
  }
  // Columns: aggregates first (leading aggregate leftmost), then per-subset
  // sorted by descending max-n_pairs. Subsets below N_MIN are dropped.
  const subsetCols = [...subsetMaxN.entries()]
    .filter(([s, n]) => n >= N_MIN && s in SUBSET_DISPLAY)
    .sort((a, b) => b[1] - a[1])
    .map(([s]) => s);
  const aggCols =
    leadingAggregate === MACRO ? [MACRO, GLOBAL] : [GLOBAL, MACRO];
  const columns = [...aggCols, ...subsetCols];

  // Sample any method that has both aggregate rows to read the per-column
  // header counts (n for global, K for macro_avg). Aggregates are constant
  // across methods (same match_groups).
  const sample = [...byMethod.values()].find(
    (m) => m.cells.has(GLOBAL) && m.cells.has(MACRO),
  );
  const globalN = sample?.cells.get(GLOBAL)?.n_pairs ?? 0;
  const macroK = sample?.cells.get(MACRO)?.n_pairs ?? 0;
  const headerLabel = (col) => {
    // Wrap the label so multi-word names ("Macro Avg") stay on a single row
    // instead of breaking across two lines when the column is tight.
    if (col === GLOBAL)
      return html`<span class="lb-col-label">Global</span><br><small>n=${globalN}</small>`;
    if (col === MACRO)
      return html`<span class="lb-col-label">Macro Avg</span><br><small>${macroK} subsets</small>`;
    return html`<span class="lb-col-label">${SUBSET_DISPLAY[col]}</span><br><small>n=${subsetMaxN.get(col)}</small>`;
  };

  let sortKey = leadingAggregate;
  const sortedMethods = () =>
    [...byMethod.values()].sort(makeComparator(sortKey));

  function colLabelText(col) {
    if (col === GLOBAL) return `Global (n=${globalN})`;
    if (col === MACRO) return `Macro Avg (${macroK} subsets)`;
    return `${SUBSET_DISPLAY[col]} (n=${subsetMaxN.get(col)})`;
  }

  function render() {
    const methods = sortedMethods();

    const root = html`<div class="lb-heatmap-wrap"></div>`;

    const table = html`<table class="lb-heatmap">
      <thead>
        <tr>
          <th class="lb-method-header">Model</th>
          ${columns.map(
            (col) => html`<th
              class=${`lb-col${col === sortKey ? " lb-col-sorted" : ""}`}
              onclick=${() => {
                sortKey = col;
                const next = render();
                root.replaceChildren(next);
              }}
              title="Click to sort by this column"
            >
              ${headerLabel(col)}
            </th>`,
          )}
        </tr>
      </thead>
      <tbody>
        ${methods.map((m) => {
          const meta = modelById.get(m.method_id);
          const family = m.family;
          const anchor = html`<a href=${`./models#${encodeURIComponent(m.method_id)}`}><code>${m.method_display}</code></a>`;
          if (meta) attachModelPopover(anchor, meta);
          const modelCell = html`<td class="lb-method">
            <span class=${`lb-family lb-family-${family}`} title=${family}></span>
            ${anchor}
          </td>`;
          return html`<tr>
            ${modelCell}
            ${columns.map((col) => {
              const c = m.cells.get(col);
              if (c == null) return html`<td class="lb-na">—</td>`;
              const bg = paColor(c.value);
              const fg = textColor(c.value);
              return html`<td
                class="lb-cell"
                style=${`background-color: ${bg}; color: ${fg};`}
              >
                ${c.value.toFixed(3)}
              </td>`;
            })}
          </tr>`;
        })}
      </tbody>
    </table>`;

    // Heatmap on the left, forest plot adjacent on the right. Both share
    // the same Y axis — explicit row height on the heatmap is what keeps
    // rows aligned (see `.lb-heatmap tbody tr { height: ... }` in
    // index.md).
    root.append(html`<div class="lb-heatmap-row">
      ${table}
      ${forestPlot(methods, sortKey, colLabelText(sortKey))}
    </div>`);
    return root;
  }

  // Top-level wrapper that swaps in re-rendered tables on column-click.
  // We return one stable node so reactive consumers don't need to re-mount.
  const initial = render();
  return initial;
}

// ---- Forest plot (PA ± SE for the currently sorted column) -----------------
//
// One row per method, in current sort order. A colored dot at the PA value
// with a horizontal whisker spanning [value − SE, value + SE] makes SE
// width explicit — Macro Avg whiskers are narrow (~0.02), per-subset
// whiskers on small-n subsets can be ~0.07-0.10.

// Side-by-side with the heatmap: the heatmap's leftmost column already
// labels the models, so the forest plot drops its left margin and shares
// the same row order/height (28px, set explicitly on `.lb-heatmap tbody tr`).
// `HEATMAP_HEADER_PX` matches the rendered height of the heatmap's <thead>
// row so the forest plot's first dot lines up with the first heatmap row.
const HEATMAP_HEADER_PX = 40;
const HEATMAP_ROW_PX = 28;

function forestPlot(methods, columnKey, columnText) {
  const visible = methods
    .map((m) => ({m, cell: m.cells.get(columnKey)}))
    .filter((x) => x.cell != null);
  if (visible.length === 0) {
    return html`<div class="lb-forest-empty">No values for ${columnText}.</div>`;
  }

  const width = 360;
  const margin = {top: HEATMAP_HEADER_PX, right: 42, bottom: 32, left: 16};
  const rowH = HEATMAP_ROW_PX;
  const height = margin.top + visible.length * rowH + margin.bottom;
  const xMin = 0.5;
  const xMax = 1.0;
  const innerW = width - margin.left - margin.right;
  const xPx = (v) =>
    margin.left + ((Math.max(xMin, Math.min(xMax, v)) - xMin) / (xMax - xMin)) * innerW;
  const xTicks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

  return svg`<svg class="lb-forest" viewBox=${`0 0 ${width} ${height}`} width=${width} style="flex: 0 0 auto;">
    ${xTicks.map(
      (t) => svg`<g>
        <line x1=${xPx(t)} x2=${xPx(t)} y1=${margin.top} y2=${height - margin.bottom}
              stroke=${t === xMin ? "#999" : "#eee"}></line>
        <text x=${xPx(t)} y=${margin.top - 8} text-anchor="middle" font-size="10" fill="#666">${t.toFixed(1)}</text>
      </g>`,
    )}
    ${visible.map(({cell}, i) => {
      const y = margin.top + i * rowH + rowH / 2;
      const cx = xPx(cell.value);
      const lo = xPx(cell.value - cell.se);
      const hi = xPx(cell.value + cell.se);
      const fill = paColor(cell.value);
      return svg`<g>
        <line x1=${lo} x2=${hi} y1=${y} y2=${y} stroke="#666" stroke-width="1"></line>
        <line x1=${lo} x2=${lo} y1=${y - 3} y2=${y + 3} stroke="#666"></line>
        <line x1=${hi} x2=${hi} y1=${y - 3} y2=${y + 3} stroke="#666"></line>
        <circle cx=${cx} cy=${y} r="4.5" fill=${fill} stroke="#333" stroke-width="0.5"></circle>
        <text x=${hi + 5} y=${y} dy="0.32em" font-size="10.5" fill="#444"
              font-variant-numeric="tabular-nums">${cell.value.toFixed(3)}</text>
      </g>`;
    })}
    <text x=${margin.left + innerW / 2} y=${height - 14} text-anchor="middle"
          font-size="10.5" font-weight="600" fill="#444">PA on ${columnText}</text>
    <text x=${margin.left + innerW / 2} y=${height - 2} text-anchor="middle"
          font-size="9.5" fill="#888">dot = value · whisker = ± SE</text>
  </svg>`;
}

// ---- Color legend ----------------------------------------------------------

export function colorLegend({width = 280, height = 16} = {}) {
  // Domain endpoints + intermediates.
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const domainMin = 0.5;
  const domainMax = 1.0;
  const x = (v) => ((v - domainMin) / (domainMax - domainMin)) * width;
  const stops = d3.range(0, 1.001, 1 / 40).map((t) => {
    const v = domainMin + t * (domainMax - domainMin);
    return {offset: `${t * 100}%`, color: paColor(v)};
  });
  return svg`<svg viewBox=${`0 0 ${width} ${height + 18}`} width=${width} style="overflow: visible;">
    <defs>
      <linearGradient id="lb-legend-grad" x1="0" x2="1">
        ${stops.map(
          (s) => svg`<stop offset=${s.offset} stop-color=${s.color}></stop>`,
        )}
      </linearGradient>
    </defs>
    <rect x="0" y="0" width=${width} height=${height} fill="url(#lb-legend-grad)" stroke="#888"></rect>
    ${ticks.map(
      (v) => svg`<g transform=${`translate(${x(v)},0)`}>
        <line y1=${height} y2=${height + 4} stroke="#666"></line>
        <text y=${height + 16} text-anchor="middle" font-size="10" fill="#555">${v.toFixed(1)}</text>
      </g>`,
    )}
  </svg>`;
}

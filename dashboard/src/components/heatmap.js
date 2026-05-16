// Leaderboard heatmap: method × subset → PA color cell.
//
// Color scale is **fixed at [0.5, 1.0]** — 0.5 (random baseline) renders
// as white, 1.0 (perfect) as deepest green. Below-random cells clamp to
// white. This makes the encoding stable across filter selections and
// across pages.

import * as d3 from "npm:d3";
import {html} from "npm:htl";

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

// Sequential interpolator on the [0.5, 1.0] domain. We skip the lightest
// 5% of `interpolateGreens` so 0.5 still has a faint green wash, making it
// distinguishable from "missing cell".
function paColor(v) {
  if (v == null) return "#ffffff";
  const t = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
  return d3.interpolateGreens(0.05 + 0.85 * t);
}

// Black/white text contrast switch — green gets darker toward the top so
// past ~0.78 we flip to white for readability.
function textColor(v) {
  if (v == null) return "#666";
  return v >= 0.78 ? "#fff" : "#000";
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
 * @param {Map}   opts.methodById metadata for each method id (for links + tooltips).
 * @param {string} opts.leadingAggregate "_macro_avg_" or "_global_" — drives the
 *   default sort + which aggregate appears leftmost.
 * @returns {HTMLElement}
 */
export function heatmap({rows, methodById, leadingAggregate = MACRO}) {
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
    if (col === GLOBAL) return html`Global<br><small>n=${globalN}</small>`;
    if (col === MACRO) return html`Macro Avg<br><small>${macroK} subsets</small>`;
    return html`${SUBSET_DISPLAY[col]}<br><small>n=${subsetMaxN.get(col)}</small>`;
  };

  let sortKey = leadingAggregate;
  const sortedMethods = () =>
    [...byMethod.values()].sort(makeComparator(sortKey));

  function render() {
    const methods = sortedMethods();

    const root = html`<div class="lb-heatmap-wrap"></div>`;

    const table = html`<table class="lb-heatmap">
      <thead>
        <tr>
          <th class="lb-method-header">Method</th>
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
          const meta = methodById.get(m.method_id);
          const family = m.family;
          const methodCell = html`<td class="lb-method">
            <span class=${`lb-family lb-family-${family}`} title=${family}></span>
            <a href=${`./methods#${encodeURIComponent(m.method_id)}`}><code>${m.method_display}</code></a>
            ${meta?.description ? html`<span class="lb-desc"> ${meta.description}</span>` : ""}
          </td>`;
          return html`<tr>
            ${methodCell}
            ${columns.map((col) => {
              const c = m.cells.get(col);
              if (c == null) {
                return html`<td class="lb-na" title="not evaluated">—</td>`;
              }
              const bg = paColor(c.value);
              const fg = textColor(c.value);
              const nLabel = col === MACRO ? `${c.n_pairs} subsets` : `n=${c.n_pairs}`;
              return html`<td
                class="lb-cell"
                style=${`background-color: ${bg}; color: ${fg};`}
                title=${`${m.method_display} · ${col === GLOBAL ? "Global" : col === MACRO ? "Macro Avg" : SUBSET_DISPLAY[col]}\nPA = ${fmt(c.value, c.se)}\n${nLabel} · ${c.n_ties} ties`}
              >
                ${c.value.toFixed(3)}
              </td>`;
            })}
          </tr>`;
        })}
      </tbody>
    </table>`;

    root.append(table);
    return root;
  }

  // Top-level wrapper that swaps in re-rendered tables on column-click.
  // We return one stable node so reactive consumers don't need to re-mount.
  const initial = render();
  return initial;
}

// ---- Color legend ----------------------------------------------------------

export function colorLegend({width = 280, height = 28} = {}) {
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const ticksWithPos = ticks.map((v) => ({
    v,
    x: ((v - 0.5) / 0.5) * width,
  }));
  const stops = d3.range(0, 1.01, 0.05).map((t) => ({
    offset: `${t * 100}%`,
    color: paColor(0.5 + t * 0.5),
  }));
  return html`<svg viewBox=${`0 0 ${width} ${height + 16}`} width=${width} style="overflow: visible;">
    <defs>
      <linearGradient id="lb-legend-grad" x1="0" x2="1">
        ${stops.map(
          (s) => html`<stop offset=${s.offset} stop-color=${s.color}></stop>`,
        )}
      </linearGradient>
    </defs>
    <rect x="0" y="0" width=${width} height=${height} fill="url(#lb-legend-grad)" stroke="#aaa"></rect>
    ${ticksWithPos.map(
      (t) => html`<g transform=${`translate(${t.x},0)`}>
        <line y1=${height} y2=${height + 4} stroke="#666"></line>
        <text y=${height + 14} text-anchor="middle" font-size="11" fill="#444">${t.v.toFixed(1)}</text>
      </g>`,
    )}
  </svg>`;
}

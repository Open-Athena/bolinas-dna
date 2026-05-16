// Methods index — one card per registered method.
//
// Each card surfaces every field from methods.yaml that would otherwise be
// buried in the rendered tooltip / table cell: training metadata, all
// outgoing links (wandb, issue, source, paper, HF, GCS), the datasets the
// method is evaluated on. Card grid is responsive (CSS grid auto-fill).

import {html} from "npm:htl";

const FAMILY_LABEL = {
  bolinas: "bolinas gLM",
  conservation: "conservation track",
  alphagenome: "AlphaGenome",
  gpn_star: "GPN-Star",
};

function paramsLabel(params) {
  if (params == null) return null;
  if (params >= 1e9) return `${(params / 1e9).toFixed(1).replace(/\.0$/, "")}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(0)}M`;
  return params.toString();
}

function link(href, label) {
  if (!href) return null;
  return html`<a href=${href} target="_blank" rel="noopener">${label}</a>`;
}

function joinLinks(parts) {
  const present = parts.filter((p) => p != null);
  if (present.length === 0) return html`<span class="muted">—</span>`;
  const out = [];
  for (let i = 0; i < present.length; i++) {
    if (i > 0) out.push(" · ");
    out.push(present[i]);
  }
  return html`${out}`;
}

function methodCard(m) {
  const training = m.training ?? {};
  const trainingFields = [
    training.data,
    paramsLabel(training.params),
    training.window_size ? `ctx ${training.window_size}` : null,
    training.objective,
  ].filter((x) => x != null);
  const checkpointLinks = [];
  if (m.checkpoint?.hf) {
    checkpointLinks.push(
      link(
        `https://huggingface.co/${m.checkpoint.hf}`,
        html`HF <code>${m.checkpoint.hf}</code>`,
      ),
    );
  }
  if (m.checkpoint?.gcs) {
    checkpointLinks.push(html`<code title="GCS checkpoint">${m.checkpoint.gcs}</code>`);
  }
  const externalLinks = [
    link(m.wandb, "wandb"),
    link(m.issue, m.experiment ? `issue #${m.experiment}` : "issue"),
    link(m.source_code, "source"),
    link(m.paper, "paper"),
  ];
  return html`<article class="method-card" id=${m.id}>
    <header class="method-card-header">
      <span class=${`method-card-family family-${m.family}`}>${FAMILY_LABEL[m.family] ?? m.family}</span>
      <h3><code>${m.display}</code></h3>
      ${m.id !== m.display ? html`<small class="method-card-step">${m.id}</small>` : ""}
    </header>
    <p class="method-card-desc">${m.description}</p>
    ${
      trainingFields.length
        ? html`<div class="method-card-row"><span class="label">Training</span><span>${trainingFields.join(" · ")}</span></div>`
        : ""
    }
    ${
      m.msa
        ? html`<div class="method-card-row"><span class="label">MSA</span><span><code>${m.msa}</code></span></div>`
        : ""
    }
    <div class="method-card-row">
      <span class="label">Datasets</span>
      <span>${m.datasets.map((d) => html`<span class="dataset-tag">${d}</span>`)}</span>
    </div>
    ${
      checkpointLinks.length
        ? html`<div class="method-card-row"><span class="label">Checkpoint</span><span>${joinLinks(checkpointLinks)}</span></div>`
        : ""
    }
    <div class="method-card-row">
      <span class="label">Links</span>
      <span>${joinLinks(externalLinks)}</span>
    </div>
  </article>`;
}

export function methodCards(methods) {
  return html`<div class="method-card-grid">${methods.map(methodCard)}</div>`;
}

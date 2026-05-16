---
title: Methods
---

# Methods

Every entry on the leaderboard, with its family, training metadata, and links out to wandb / source / HF / GCS / tracking issues. This is Phase B — the cards UI ships in Phase C.

```js
const methods = await FileAttachment("data/methods.json").json();
```

```js
const rows = methods.map(m => ({
  id: m.id,
  display: m.display,
  family: m.family,
  description: m.description,
  experiment: m.experiment ?? "",
  wandb: m.wandb ?? "",
  issue: m.issue ?? "",
  checkpoint: m.checkpoint?.gcs ?? (m.checkpoint?.hf ? `hf://${m.checkpoint.hf}` : ""),
  datasets: m.datasets.join(", "),
}));
```

```js
Inputs.table(rows, {
  rows: 50,
  width: {
    id: 220,
    display: 180,
    family: 110,
    description: 200,
    experiment: 90,
    wandb: 80,
    issue: 80,
    checkpoint: 260,
    datasets: 200,
  },
  format: {
    wandb: u => u ? html`<a href=${u} target="_blank">run</a>` : "",
    issue: u => u ? html`<a href=${u} target="_blank">issue</a>` : "",
  },
})
```

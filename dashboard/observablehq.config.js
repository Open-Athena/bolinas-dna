// Observable Framework site config for the Bolinas-DNA leaderboard.
// See: https://observablehq.com/framework/config

export default {
  title: "Bolinas-DNA Leaderboard",
  root: "src",
  output: "dist",

  // Sidebar navigation. v1 ships Mendelian only; complex_traits / eqtl
  // will be added later (see the dashboard plan).
  pages: [
    {name: "Mendelian", path: "/"},
    {name: "Methods", path: "/methods"},
    {name: "About", path: "/about"},
  ],

  // Python data loaders run via `uv run python` so they pick up the project
  // venv (polars + boto3 + the local `bolinas` package).
  interpreters: {
    ".py": ["uv", "run", "python"],
  },

  // Suppress Observable's automatic header (we render the page title in
  // the markdown body instead, alongside dataset metadata).
  header: "",
  footer: ({path}) =>
    `Source: <a href="https://github.com/Open-Athena/bolinas-dna/blob/main/dashboard/src${path}.md">dashboard/src${path}.md</a> · <a href="https://github.com/Open-Athena/bolinas-dna/blob/main/dashboard/methods.yaml">methods.yaml</a>`,
};

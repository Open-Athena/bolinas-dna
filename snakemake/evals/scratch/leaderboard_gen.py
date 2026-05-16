"""Generate leaderboard markdown tables for issues #161 / #162 / #172.

Pulls per-(method, dataset, subset) PairwiseAccuracy + SE from S3:
  - conservation_eval: 7 conservation tracks
  - evals_v2: 5 model checkpoints
  - alphagenome_eval: AlphaGenome variant scorer
  - gpn_star_eval: GPN-Star V/M/P (calibrated variants; predictions
    scored externally by TraitGym)

Combines into one table per dataset. n_pairs ≥ 30 cutoff for per-subset
columns. Two aggregate columns are prepended:

  - Global: PA across ALL pairs (no n filter). Sourced from the `_global_`
    row written by `compute_pairwise_metrics`.
  - Macro Avg: unweighted mean of per-subset PAs over n≥30 subsets. Sourced
    from the `_macro_avg_` row.

Methods are sorted by `Global` PA descending so the best-overall method
appears at the top of each table.

Bolding rule (top method per column, plus any within 0.01 of the top) applies
to every column including the aggregates.

Outputs three markdown chunks ready to drop into the leaderboard issues. With
`--patch-issues`, surgically PATCHes the bodies of #161/#162/#172 instead of
relying on manual paste.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import NamedTuple

import polars as pl
import yaml

from bolinas.pipelines.evals.gpn_star import GPN_STAR_MODELS, GPN_STAR_SCORE_COLUMN
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET

# Per-dataset score_type per pipeline.
SCORE_TYPE = {
    "evals_v2": {
        "mendelian_traits": "minus_llr",
        "complex_traits": "abs_llr",
        "eqtl": "abs_llr",
    },
    "conservation": "score",
    "alphagenome": "alphagenome_max_l2",
    # GPN-Star ships with both calibrated and uncalibrated variants in the
    # metrics parquet; the leaderboard renders the calibrated one (see
    # `bolinas.pipelines.evals.gpn_star.GPN_STAR_SCORE_COLUMN`).
    "gpn_star": GPN_STAR_SCORE_COLUMN,
}


class EvalsV2Method(NamedTuple):
    """One row of the evals_v2 section of the leaderboard.

    ``parquet`` is the model name as it appears under
    ``s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/{parquet}/{dataset}.parquet``
    — same string as the ``name`` field in `snakemake/analysis/evals_v2/config/config.yaml`.

    ``display`` is what's shown in the leaderboard ``method`` column. For the
    legacy single-step entries it equals ``parquet``; for the newer per-step
    entries (e.g. ``exp21-promoters-yolo-step-22000``) it's stripped to the
    run's short name (``exp21-promoters-yolo``) — the step lives in the
    Reproducibility section.

    ``datasets`` are the datasets this entry has metrics for. Older entries
    span all three; entries added with ``datasets: [mendelian_traits]`` in
    the evals_v2 config only show up in the mendelian leaderboard.
    """

    parquet: str
    display: str
    comment: str | None
    datasets: tuple[str, ...] = ("mendelian_traits", "complex_traits", "eqtl")


EVALS_V2_MODELS: list[EvalsV2Method] = [
    # Legacy single-step entries — evaluated on all 3 leaderboard datasets.
    EvalsV2Method("exp55-mammals", "exp55-mammals", "promoters, mammals"),
    EvalsV2Method("exp58-mammals", "exp58-mammals", "CDS, mammals"),
    EvalsV2Method("exp58-animals", "exp58-animals", "CDS, animals"),
    EvalsV2Method("exp59-mammals", "exp59-mammals", "downstream, mammals"),
    EvalsV2Method("exp136-proj_v30", "exp136-proj_v30", "enhancers, mammals"),
    # Older `exp166-p1B` (step-16398, HF) retained for #162/#172 only — its
    # mendelian slot in #161 was replaced by `exp166-v0.1-p1B` (below) on
    # 2026-05-16. Re-include here on complex/eqtl until v0.1 is also scored
    # on those datasets.
    EvalsV2Method(
        "exp166-p1B",
        "exp166-p1B",
        "zoonomia, generalist, 1B",
        datasets=("complex_traits", "eqtl"),
    ),
    # Convergence-style entries (final checkpoint only here per
    # "last checkpoint per run" convention). All scored on mendelian only
    # in the evals_v2 config, so they appear only in #161.
    EvalsV2Method(
        "exp21-promoters-yolo-step-22000",
        "exp21-promoters-yolo",
        "promoters, animals, 1.7B",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp13-mixture-equal-step-26000",
        "exp13-mixture-equal",
        "50/50 promoter+CDS, animals, 1.7B",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp13-mixture-proportional-step-26000",
        "exp13-mixture-proportional",
        "10/90 promoter+CDS, animals, 1.7B",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp27-cds-yolo-step-34000",
        "exp27-cds-yolo",
        "CDS, animals, 1.7B",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp55-humans-step-16999",
        "exp55-humans",
        "promoters, humans",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp55-primates-step-16999",
        "exp55-primates",
        "promoters, primates",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp55-vertebrates-step-16999",
        "exp55-vertebrates",
        "promoters, vertebrates",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp55-animals-step-16999",
        "exp55-animals",
        "promoters, animals",
        datasets=("mendelian_traits",),
    ),
    EvalsV2Method(
        "exp58-vertebrates-step-16999",
        "exp58-vertebrates",
        "CDS, vertebrates",
        datasets=("mendelian_traits",),
    ),
    # Replaces the legacy `exp166-p1B` HF entry (step-16398) — that run has
    # been superseded by this v0.1 c127da run at step-27329 (1.65× more steps).
    EvalsV2Method(
        "exp166-v0.1-p1B-step-27329",
        "exp166-v0.1-p1B",
        "zoonomia, generalist, 1B, v0.1",
        datasets=("mendelian_traits",),
    ),
]
CONSERVATION_TRACKS = [
    "phastCons_100v",
    "phastCons_43p",
    "phastCons_470m",
    "phyloP_100v",
    "phyloP_241m",
    "phyloP_447m",
    "phyloP_470m",
]

SUBSET_DISPLAY = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5' UTR",
    "distal": "Distal",
    "3_prime_UTR_variant": "3' UTR",
    "tss_proximal": "Promoter",
    "non_coding_transcript_exon_variant": "ncRNA",
    "synonymous_variant": "Synonymous",
}

DATASETS = ("mendelian_traits", "complex_traits", "eqtl")
DATASET_ISSUE = {
    "mendelian_traits": 161,
    "complex_traits": 162,
    "eqtl": 172,
}

# Which aggregate is the "headline" for each dataset — controls (a) the sort
# axis (top of table = top method on this aggregate) and (b) which of the two
# aggregate columns appears leftmost. Mendelian uses Macro Avg because the
# variant composition is ~92% missense — a ClinVar annotator-history artifact,
# not pathogenicity reality — so Global PA over-weights methods specialized
# for protein-coding variant interpretation. Complex / eqtl have very different
# subset compositions where the same bias doesn't apply, so they stay on Global.
LEADING_AGGREGATE = {
    "mendelian_traits": "macro",
    "complex_traits": "global",
    "eqtl": "global",
}
S3 = "s3://oa-bolinas"
SPLIT = "train"
N_MIN = 30
REPO = "Open-Athena/bolinas-dna"


def fmt(value: float, se: float) -> str:
    return f"{value:.3f} ± {se:.3f}"


def gather_methods(dataset: str) -> list[tuple[str, str | None, pl.DataFrame]]:
    """Return [(method_name, comment, full_df), ...] for one dataset.

    full_df includes both per-subset rows and the aggregate `_global_` /
    `_macro_avg_` rows."""
    rows: list[tuple[str, str | None, pl.DataFrame]] = []

    # 1. conservation tracks
    cons = pl.read_parquet(
        f"{S3}/snakemake/conservation_eval/results/{dataset}/metrics_{SPLIT}.parquet"
    )
    for track in CONSERVATION_TRACKS:
        df = cons.filter(pl.col("score_name") == track).select(
            ["subset", "value", "se", "n_pairs"]
        )
        rows.append((f"`{track}`", None, df))

    # 2. evals_v2 models — skip entries that weren't scored on this dataset
    # (per-(model, dataset) inclusion). The legacy single-step entries cover
    # all 3 datasets; convergence-style entries are mendelian-only.
    sct = SCORE_TYPE["evals_v2"][dataset]
    for entry in EVALS_V2_MODELS:
        if dataset not in entry.datasets:
            continue
        df = pl.read_parquet(
            f"{S3}/snakemake/analysis/evals_v2/results/metrics/{entry.parquet}/{dataset}.parquet"
        )
        df = df.filter(pl.col("score_type") == sct).filter(pl.col("split") == SPLIT)
        df = df.select(["subset", "value", "se", "n_pairs"])
        rows.append((f"`{entry.display}`", entry.comment, df))

    # 3. alphagenome
    try:
        ag = pl.read_parquet(
            f"{S3}/snakemake/alphagenome_eval/results/metrics/{dataset}.parquet"
        )
        ag = ag.filter(pl.col("score_type") == SCORE_TYPE["alphagenome"]).filter(
            pl.col("split") == SPLIT
        )
        rows.append(
            (
                "`AlphaGenome`",
                "variant scorer, API",
                ag.select(["subset", "value", "se", "n_pairs"]),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! alphagenome metrics missing for {dataset}: {exc}")

    # 4. gpn_star — calibrated variants only (uncalibrated numbers live in
    # the #145 eval comment for reference). One row per model V/M/P. Catch
    # only the S3 read so a present-but-malformed parquet fails loud rather
    # than getting silently treated as "metrics missing".
    try:
        gs = pl.read_parquet(
            f"{S3}/snakemake/gpn_star_eval/results/metrics/{dataset}.parquet"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! gpn_star metrics missing for {dataset}: {exc}")
        gs = None
    if gs is not None:
        gs = gs.filter(pl.col("score_type") == SCORE_TYPE["gpn_star"][dataset]).filter(
            pl.col("split") == SPLIT
        )
        for model in GPN_STAR_MODELS:
            df = gs.filter(pl.col("model") == f"GPN-Star-{model}").select(
                ["subset", "value", "se", "n_pairs"]
            )
            assert df.height > 0, (
                f"no GPN-Star-{model} rows for {dataset!r} in metrics parquet"
            )
            rows.append((f"`GPN-Star-{model}`", None, df))

    return rows


def _split_method(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[float, float, int], tuple[float, float, int]]:
    """Split a method's metrics frame into per-subset rows, _global_, and
    _macro_avg_ aggregate tuples (value, se, n)."""
    per_sub = df.filter(~pl.col("subset").is_in([GLOBAL_SUBSET, MACRO_AVG_SUBSET]))
    g = df.filter(pl.col("subset") == GLOBAL_SUBSET)
    m = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert g.height == 1 and m.height == 1, (
        f"expected one _global_ and one _macro_avg_ row, got {g.height}/{m.height}"
    )
    g_tup = (g[0, "value"], g[0, "se"], int(g[0, "n_pairs"]))
    m_tup = (m[0, "value"], m[0, "se"], int(m[0, "n_pairs"]))
    return per_sub, g_tup, m_tup


def build_table(dataset: str) -> str:
    rows = gather_methods(dataset)

    # Pre-split each method once: per-subset rows + the two aggregate tuples.
    methods = [(method, comment, *_split_method(df)) for method, comment, df in rows]

    # Sort descending on the leading aggregate (Global for most datasets,
    # Macro Avg for mendelian — see LEADING_AGGREGATE for the rationale).
    # Python's sort is stable, so ties keep insertion order (conservation →
    # evals_v2 → AlphaGenome → GPN-Star).
    leading = LEADING_AGGREGATE[dataset]
    sort_idx = 3 if leading == "global" else 4  # 3 = global tuple, 4 = macro tuple
    methods.sort(key=lambda m: -m[sort_idx][0])

    subset_n: dict[str, int] = {}
    for _, _, per_sub, _, _ in methods:
        for s, n in per_sub.select(["subset", "n_pairs"]).iter_rows():
            subset_n[s] = max(subset_n.get(s, 0), int(n))
    subsets = [
        s
        for s, n in sorted(subset_n.items(), key=lambda kv: -kv[1])
        if n >= N_MIN and s in SUBSET_DISPLAY
    ]
    if not subsets:
        return f"# {dataset}\n\nNo subset has n_pairs ≥ {N_MIN}.\n"

    # Per-subset cell values + per-column top (for bolding).
    cell_pa: dict[tuple[str, str], tuple[float, float]] = {}
    top_subset: dict[str, float] = {}
    for method, _, per_sub, _, _ in methods:
        for s, v, se, _ in per_sub.iter_rows():
            if s in subsets:
                cell_pa[(method, s)] = (v, se)
                top_subset[s] = max(top_subset.get(s, -1.0), v)

    top_global = max(g[0] for _, _, _, g, _ in methods)
    top_macro = max(m[0] for _, _, _, _, m in methods)

    # Aggregate-column counts are constant across methods (same match_groups).
    _, _, _, (_, _, global_n), (_, _, macro_k) = methods[0]

    # Column order: leading aggregate first, secondary aggregate after — so
    # the leftmost cell of the top row matches the sort axis.
    global_header = f"Global<br>(n={global_n})"
    macro_header = f"Macro Avg<br>({macro_k} subsets)"
    aggregate_headers = (
        [macro_header, global_header]
        if leading == "macro"
        else [global_header, macro_header]
    )
    header_cols = aggregate_headers + [
        f"{SUBSET_DISPLAY[s]}<br>(n={subset_n[s]})" for s in subsets
    ]
    header = "| method | " + " | ".join(header_cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(header_cols)) + "|"
    lines = [header, sep]

    for method, comment, _, (gv, gse, _), (mv, mse, _) in methods:
        label = method + (f" ({comment})" if comment else "")
        global_cell = f"**{fmt(gv, gse)}**" if gv >= top_global - 0.01 else fmt(gv, gse)
        macro_cell = f"**{fmt(mv, mse)}**" if mv >= top_macro - 0.01 else fmt(mv, mse)
        cells = (
            [macro_cell, global_cell]
            if leading == "macro"
            else [global_cell, macro_cell]
        )
        for s in subsets:
            if (method, s) not in cell_pa:
                cells.append("—")
                continue
            v, se = cell_pa[(method, s)]
            text = fmt(v, se)
            if v >= top_subset[s] - 0.01:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---- Issue body PATCH ------------------------------------------------------


def _gh_get_issue_body(issue_number: int) -> str:
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/{issue_number}", "--jq", ".body"],
        check=True,
        capture_output=True,
        text=True,
    )
    # The jq filter strips the surrounding JSON quoting but the trailing
    # newline from subprocess capture is artifactual — issue bodies do not
    # end with a newline themselves.
    return out.stdout.rstrip("\n")


def _gh_patch_issue_body(issue_number: int, body: str) -> None:
    """PATCH /repos/.../issues/<n> via gh api with JSON stdin so multi-line
    bodies aren't subject to shell-quoting or form-encoding fragility."""
    subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "PATCH",
            f"repos/{REPO}/issues/{issue_number}",
            "--input",
            "-",
        ],
        input=json.dumps({"body": body}),
        check=True,
        capture_output=True,
        text=True,
    )


# The leaderboard table is the only contiguous block of pipe-delimited lines
# inside the Results section. We scope the replacement to that section (between
# the ## Results header and the following horizontal rule) so the Reproducibility
# table in <details> — which also begins with "| method |" — can never be
# accidentally targeted, even if section order changes in future edits.
RESULTS_SECTION_RE = re.compile(
    r"(## Results — train split\n.*?)(\n---\n)",
    re.DOTALL,
)
TABLE_RE = re.compile(
    r"\| method \|[^\n]*\n(?:\|[^\n]*\n)+",
    re.MULTILINE,
)


def _replace_table(body: str, new_table: str) -> str:
    section_match = RESULTS_SECTION_RE.search(body)
    if section_match is None:
        raise RuntimeError("could not find Results section in issue body")
    section, sep = section_match.group(1), section_match.group(2)
    if not TABLE_RE.search(section):
        raise RuntimeError("could not find leaderboard table in Results section")
    new_section = TABLE_RE.sub(new_table + "\n", section, count=1)
    return (
        body[: section_match.start()] + new_section + sep + body[section_match.end() :]
    )


def _idempotent_replace(
    body: str, old_variants: list[str], new_text: str, what: str
) -> str:
    """Replace the first matching variant in ``old_variants`` with ``new_text``,
    or no-op if ``new_text`` is already present (prior PATCH). Raises if
    neither anchor is found — fails loud so a future schema drift can't
    silently skip the update."""
    for old in old_variants:
        if old in body:
            return body.replace(old, new_text, 1)
    if new_text in body:
        return body
    raise RuntimeError(f"could not locate {what}")


def _update_intro_paragraph(
    body: str, global_n: int, macro_k: int, dataset: str
) -> str:
    leading = LEADING_AGGREGATE[dataset]
    if leading == "macro":
        new_intro = (
            f"PairwiseAccuracy ± SE per method. Higher is better. "
            f"**Sorted by Macro Avg** for this dataset: the variant "
            f"composition skews heavily toward missense (~92% of pairs) "
            f"due to ClinVar annotator history, not pathogenicity reality, "
            f"so Global PA over-weights methods specialized for "
            f"protein-coding variant interpretation; Macro Avg gives equal "
            f"weight to each consequence subset. Two leftmost columns "
            f"aggregate across the per-subset cells: **Macro Avg** = "
            f"unweighted mean of per-subset PAs over the {macro_k} "
            f"reported subsets; **Global** = PA across **all** matched "
            f"pairs (n={global_n}), including any subsets below the "
            f"threshold. Subsets with `n_pairs < 30` are excluded from "
            f"the per-subset columns by convention (held across "
            f"leaderboard datasets in this repo)."
        )
    else:
        new_intro = (
            f"PairwiseAccuracy ± SE per method. Higher is better. Two "
            f"leftmost columns aggregate across the per-subset cells: "
            f"**Global** = PA across **all** matched pairs (n={global_n}), "
            f"including any subsets below the threshold; **Macro Avg** = "
            f"unweighted mean of per-subset PAs over the {macro_k} "
            f"reported subsets. Subsets with `n_pairs < 30` are excluded "
            f"from the per-subset columns by convention (held across "
            f"leaderboard datasets in this repo)."
        )
    old_variants = [
        # Mendelian / eqtl phrasing.
        "PairwiseAccuracy ± SE per consequence subset. Higher is better. "
        "Subsets with `n_pairs < 30` are excluded by convention (held across "
        "leaderboard datasets in this repo).",
        # Complex traits phrasing (extra "Most consequence subsets…" sentence).
        "PairwiseAccuracy ± SE per consequence subset. Higher is better. "
        "Subsets with `n_pairs < 30` are excluded by convention (held across "
        "leaderboard datasets in this repo). Most consequence subsets in "
        "this dataset fall below that threshold — see Sizes for full breakdown.",
        # Prior global-sort phrasing (in case we're switching from global → macro).
        f"PairwiseAccuracy ± SE per method. Higher is better. Two leftmost "
        f"columns aggregate across the per-subset cells: **Global** = PA "
        f"across **all** matched pairs (n={global_n}), including any "
        f"subsets below the threshold; **Macro Avg** = unweighted mean of "
        f"per-subset PAs over the {macro_k} reported subsets. Subsets with "
        f"`n_pairs < 30` are excluded from the per-subset columns by "
        f"convention (held across leaderboard datasets in this repo).",
    ]
    return _idempotent_replace(body, old_variants, new_intro, "intro paragraph")


def _update_bold_rule(body: str) -> str:
    old = (
        "**Bold** = top method per subset, plus any method within **0.01** "
        "PairwiseAccuracy of the top."
    )
    new = (
        "**Bold** = top method per column (including aggregates), plus any "
        "method within **0.01** PairwiseAccuracy of the top."
    )
    return _idempotent_replace(body, [old], new, "bold-rule sentence")


def _update_glm_protocol(body: str) -> str:
    """Reflect that evals_v2 gLMs now run FWD + RC strand averaging
    (was forward-strand only). Also rewrites the GPN-Star calibration
    row's parenthetical and the NaN-handling note."""
    body = _idempotent_replace(
        body,
        ["| gLM with causal LM head (LLR) |"],
        "| gLM with causal LM head (LLR), FWD + RC averaged |",
        "gLM-row kind column",
    )
    body = _idempotent_replace(
        body,
        [
            "the other gLM rows here are forward-strand only (RC averaging is a "
            "planned addition)"
        ],
        "the other gLM rows here are also FWD + RC averaged (per #175 conclusion 2)",
        "GPN-Star RC-parenthetical",
    )
    body = _idempotent_replace(
        body,
        ["- **gLM (LLR):** no NaN expected — every variant gets a forward pass."],
        "- **gLM (LLR):** no NaN expected — every variant gets a forward + reverse-complement strand pass; results averaged element-wise.",
        "gLM NaN-handling note",
    )
    return body


EVALS_V2_CONFIG = (
    Path(__file__).resolve().parents[2] / "analysis/evals_v2/config/config.yaml"
)


def _load_evals_v2_sources() -> dict[str, str]:
    """Map model `name` → its `gcs_path` (or `hf://<repo>` shorthand)
    by reading `snakemake/analysis/evals_v2/config/config.yaml`. Single
    source of truth for Reproducibility-section regeneration."""
    cfg = yaml.safe_load(EVALS_V2_CONFIG.read_text())
    sources: dict[str, str] = {}
    for m in cfg["models"]:
        if "gcs_path" in m:
            sources[m["name"]] = m["gcs_path"]
        elif "hf_repo" in m:
            sources[m["name"]] = f"hf://{m['hf_repo']}"
        else:
            raise RuntimeError(f"model {m['name']!r} has neither gcs_path nor hf_repo")
    return sources


def _render_glm_repro_row(entry: "EvalsV2Method", source: str, dataset: str) -> str:
    """One row of the Reproducibility table for an evals_v2 gLM entry."""
    if source.startswith("gs://"):
        m = re.search(r"checkpoints/([^/]+)/hf/step-(\d+)", source)
        if m is None:
            raise RuntimeError(
                f"unexpected gcs_path shape for {entry.parquet}: {source}"
            )
        run_name, step = m.group(1), m.group(2)
        source_md = f"`{source}`"
        wandb_md = f"[run](https://wandb.ai/gonzalobenegas/marin/runs/{run_name})"
    elif source.startswith("hf://"):
        repo = source.removeprefix("hf://")
        step_m = re.search(r"step-(\d+)", repo)
        step = step_m.group(1) if step_m else "–"
        source_md = f"HF Hub [`{repo}`](https://huggingface.co/{repo})"
        wandb_md = "–"
    else:
        raise RuntimeError(f"unrecognized source URI for {entry.parquet}: {source!r}")
    parquet = (
        f"`s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
        f"{entry.parquet}/{dataset}.parquet`"
    )
    return f"| `{entry.display}` | {step} | {source_md} | {wandb_md} | {parquet} |"


# The Repro section's gLM block = a contiguous run of rows starting with
# `| \`exp...\` |`. We don't anchor on specific entry names so the regex
# survives EVALS_V2_MODELS edits. (The conservation / GPN-Star / AlphaGenome
# rows start with different method prefixes so they don't match.)
REPRO_GLM_BLOCK_RE = re.compile(
    r"(?P<block>(?:\| `exp[^`]+` \| [^\n]+\n)+)",
    re.MULTILINE,
)


def _update_repro_glm_block(body: str, dataset: str) -> str:
    """Regenerate the gLM rows of the Reproducibility table from
    EVALS_V2_MODELS. Preserves the surrounding static rows (conservation,
    GPN-Star, AlphaGenome) untouched."""
    sources = _load_evals_v2_sources()
    new_rows = [
        _render_glm_repro_row(entry, sources[entry.parquet], dataset)
        for entry in EVALS_V2_MODELS
        if dataset in entry.datasets and entry.parquet in sources
    ]
    if not new_rows:
        return body
    new_block = "\n".join(new_rows) + "\n"
    match = REPRO_GLM_BLOCK_RE.search(body)
    if match is None:
        raise RuntimeError("could not find gLM block in Reproducibility section")
    return body[: match.start()] + new_block + body[match.end() :]


def _bump_last_updated(body: str, today: str) -> str:
    return re.sub(
        r"\*\*Last updated:\*\* [0-9]{4}-[0-9]{2}-[0-9]{2}",
        f"**Last updated:** {today}",
        body,
        count=1,
    )


def _prepend_changelog_entry(body: str, entry: str) -> str:
    """Insert a new changelog item at the top of the Changelog <details>
    block. Idempotent: if the entry text already appears verbatim, no-op."""
    anchor = (
        "Protocol or data changes go below. The body above always reflects "
        "the *current* protocol; this section is the audit trail."
    )
    if anchor not in body:
        raise RuntimeError("missing changelog anchor")
    if entry in body:
        return body
    return body.replace(
        anchor + "\n\n",
        anchor + "\n\n" + entry + "\n\n",
        1,
    )


def patch_issue(dataset: str, table_md: str) -> None:
    issue_number = DATASET_ISSUE[dataset]
    print(f"  ↳ fetching issue #{issue_number} body...")
    body = _gh_get_issue_body(issue_number)

    # Pull global_n / macro_k out of the rendered table for the explainer.
    m_global = re.search(r"Global<br>\(n=(\d+)\)", table_md)
    m_macro = re.search(r"Macro Avg<br>\((\d+) subsets\)", table_md)
    assert m_global and m_macro, "could not parse aggregate-column counts"
    global_n = int(m_global.group(1))
    macro_k = int(m_macro.group(1))

    new_body = _replace_table(body, table_md)
    new_body = _update_intro_paragraph(new_body, global_n, macro_k, dataset)
    new_body = _update_bold_rule(new_body)
    new_body = _update_glm_protocol(new_body)
    new_body = _update_repro_glm_block(new_body, dataset)
    new_body = _bump_last_updated(new_body, str(date.today()))
    if dataset == "mendelian_traits":
        new_body = _prepend_changelog_entry(
            new_body,
            "- **2026-05-16** — sort axis switched from Global PA to "
            "**Macro Avg** for this leaderboard only. Rationale: the "
            "variant composition is ~92% missense (a ClinVar annotator-"
            "history artifact, not pathogenicity reality), so Global PA "
            "over-weights methods specialized for protein-coding variant "
            "interpretation. Macro Avg gives equal weight to each "
            "consequence subset. Column order updated so the leading "
            "aggregate (Macro Avg) is leftmost. #162 / #172 continue to "
            "sort by Global PA.",
        )
        new_body = _prepend_changelog_entry(
            new_body,
            "- **2026-05-16** — added 10 evals_v2 gLM rows (final-checkpoint, "
            "mendelian only): `exp21-promoters-yolo` (#21), "
            "`exp13-mixture-equal` / `exp13-mixture-proportional` (#13), "
            "`exp27-cds-yolo` (#27), `exp55-{humans, primates, vertebrates, "
            "animals}` (#55, promoters across evolutionary timescales), "
            "`exp58-vertebrates` (#58, CDS across timescales). Replaced "
            "legacy `exp166-p1B` (step-16398) with `exp166-v0.1-p1B` "
            "(step-27329, the c127da run that just finished — 1.65× more "
            "steps; Global PA +0.65 pp, Macro Avg −2.76 pp vs the old).",
        )
    new_body = _prepend_changelog_entry(
        new_body,
        "- **2026-05-14** — added `exp166-p1B` (1B zoonomia-v1-v1 generalist; "
        "HF: [`bolinas-dna/exp166-p1B-step-16398`](https://huggingface.co/bolinas-dna/exp166-p1B-step-16398)). "
        "All 5 existing `evals_v2` gLM rows re-scored under FWD + RC strand "
        "averaging (per #175 conclusion 2; previously forward-strand only); "
        "per-cell PA shifts up to ~0.06. Variant-score kernel slimmed to "
        "LLR + per-position next-token JSD (drops the `embed_*_l2` columns; "
        "JSD has Spearman ρ ≈ 0.90 with `embed_last_l2` within mendelian "
        "subsets per #175 conclusion 9). Leaderboard score column unchanged.",
    )
    # Hardcoded dates on changelog entries (matching the date the change
    # actually landed) so re-runs of this script don't re-add the entry with
    # today's date each time — idempotency relies on the entry text being
    # stable across runs.
    new_body = _prepend_changelog_entry(
        new_body,
        f"- **2026-05-12** — added `Global` (PA across all pairs, "
        f"n={global_n}) and `Macro Avg` (unweighted PA over n≥30 subsets, "
        f"{macro_k} subsets) aggregate columns. Implemented as new "
        f"`_global_` / `_macro_avg_` rows in `compute_pairwise_metrics` "
        f"(`src/bolinas/evals/metrics.py`); all 3 metric pipelines re-run.",
    )
    calibrated = SCORE_TYPE["gpn_star"][dataset]
    raw = calibrated.replace("_calibrated", "")
    new_body = _prepend_changelog_entry(
        new_body,
        f"- **2026-05-13** — added `GPN-Star-V`, `GPN-Star-M`, `GPN-Star-P` "
        f"(calibrated variants only; score column `{calibrated}`). Source "
        f"predictions: [#145 comment](https://github.com/Open-Athena/bolinas-dna/issues/145#issuecomment-4444680280); "
        f"evaluation + uncalibrated `{raw}` numbers: "
        f"[#145 comment](https://github.com/Open-Athena/bolinas-dna/issues/145#issuecomment-4444856709). "
        f"Pipeline = TraitGym at commit `05135727`. Existing rows re-sorted "
        f"by `Global` PA descending (top of table = current best) — values "
        f"unchanged.",
    )

    if new_body == body:
        print(f"  ↳ no changes for issue #{issue_number}")
        return
    print(f"  ↳ patching issue #{issue_number}...")
    _gh_patch_issue_body(issue_number, new_body)
    print(f"  ↳ patched #{issue_number}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-issues",
        action="store_true",
        help="After printing, surgically PATCH the bodies of #161/#162/#172.",
    )
    parser.add_argument(
        "--patch-issue",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help=(
            "PATCH a specific leaderboard issue (one of 161, 162, 172). "
            "Repeatable. Useful when only some datasets' tables changed."
        ),
    )
    args = parser.parse_args()

    issue_to_dataset = {n: ds for ds, n in DATASET_ISSUE.items()}
    if args.patch_issues:
        datasets_to_patch = list(DATASETS)
    else:
        bad = [n for n in args.patch_issue if n not in issue_to_dataset]
        if bad:
            parser.error(
                f"--patch-issue must be one of {sorted(issue_to_dataset)}, got {bad}"
            )
        datasets_to_patch = [issue_to_dataset[n] for n in args.patch_issue]

    tables: dict[str, str] = {}
    for ds in DATASETS:
        print(f"\n{'#' * 70}\n# {ds}\n{'#' * 70}\n")
        table = build_table(ds)
        print(table)
        tables[ds] = table

    if datasets_to_patch:
        print("\n" + "#" * 70)
        print("# Patching GitHub issues")
        print("#" * 70)
        for ds in datasets_to_patch:
            patch_issue(ds, tables[ds])


if __name__ == "__main__":
    main()

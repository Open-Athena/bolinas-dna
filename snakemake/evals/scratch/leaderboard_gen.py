"""Generate leaderboard markdown tables for issues #161 / #162 / #172 and
optionally PATCH the issue bodies in-place.

This script is a thin CLI on top of ``bolinas.pipelines.evals.leaderboard``
(which holds the metric aggregation + markdown rendering) and
``bolinas.pipelines.evals.models`` (which loads the method registry from
``dashboard/models.yaml``). It exists for the transition window while the
issue-body tables are still live; once those issues are closed in favour of
the dashboard, this script can be deleted.

The body-rewrite helpers below are issue-body-specific (intro paragraph
phrasing, reproducibility table, changelog entries) and stay here rather than
moving into the library.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import yaml

from bolinas.pipelines.evals.leaderboard import (
    DATASET_ISSUE,
    LEADING_AGGREGATE,
    build_table,
    score_type_for,
)
from bolinas.pipelines.evals.models import Model, models_for_dataset

DATASETS: tuple[str, ...] = ("mendelian_traits", "complex_traits", "eqtl")
REPO = "Open-Athena/bolinas-dna"

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


@functools.cache
def _load_evals_v2_config_models() -> dict[str, dict]:
    """Map evals_v2 model `name` → its raw config dict.

    Single source of truth for the cross-check that every bolinas entry in
    `dashboard/models.yaml` is actually scored by the pipeline."""
    cfg = yaml.safe_load(EVALS_V2_CONFIG.read_text())
    return {m["name"]: m for m in cfg["models"]}


def _model_source_uri(method: Model) -> str:
    """Return the canonical source URI for a bolinas method's checkpoint:
    its GCS path if present, else ``hf://<repo>``."""
    assert method.family == "bolinas", method.id
    assert method.checkpoint is not None, method.id
    if method.checkpoint.gcs:
        return method.checkpoint.gcs
    assert method.checkpoint.hf, method.id
    return f"hf://{method.checkpoint.hf}"


def _wandb_run_name(method: Model) -> str | None:
    """Extract the wandb run name from a GCS checkpoint path of the shape
    ``gs://…/checkpoints/<run_name>/hf/step-<N>``. Returns None when the
    checkpoint is HF-hosted (no wandb mirror in that case)."""
    if method.checkpoint is None or not method.checkpoint.gcs:
        return None
    m = re.search(r"checkpoints/([^/]+)/hf/step-\d+", method.checkpoint.gcs)
    return m.group(1) if m else None


def _render_glm_repro_row(method: Model, dataset: str) -> str:
    """One row of the Reproducibility table for a bolinas method entry."""
    source = _model_source_uri(method)
    if source.startswith("gs://"):
        m = re.search(r"checkpoints/([^/]+)/hf/step-(\d+)", source)
        if m is None:
            raise RuntimeError(f"unexpected gcs_path shape for {method.id}: {source}")
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
        raise RuntimeError(f"unrecognized source URI for {method.id}: {source!r}")
    parquet = (
        f"`s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
        f"{method.id}/{dataset}.parquet`"
    )
    return f"| `{method.display}` | {step} | {source_md} | {wandb_md} | {parquet} |"


# Stable HTML-comment anchors around the auto-regenerated gLM block.
# GitHub-flavored markdown hides HTML comments in rendered output, so these
# are invisible to readers but unmissable for the regex. On first run against
# a body that pre-dates the anchors, we fall back to the legacy `exp`-prefix
# regex and emit the anchors so future runs use the stable markers.
REPRO_GLM_ANCHOR_START = "<!-- evals_v2-glm-rows:start -->"
REPRO_GLM_ANCHOR_END = "<!-- evals_v2-glm-rows:end -->"
REPRO_GLM_ANCHORED_RE = re.compile(
    re.escape(REPRO_GLM_ANCHOR_START) + r"\n.*?\n" + re.escape(REPRO_GLM_ANCHOR_END),
    re.DOTALL,
)
REPRO_GLM_LEGACY_RE = re.compile(
    r"(?:\| `exp[^`]+` \| [^\n]+\n)+",
    re.MULTILINE,
)


def _update_repro_glm_block(body: str, dataset: str) -> str:
    """Regenerate the gLM rows of the Reproducibility table from models.yaml.
    Preserves the surrounding static rows (conservation, GPN-Star, AlphaGenome)
    untouched."""
    new_rows = [
        _render_glm_repro_row(m, dataset)
        for m in models_for_dataset(dataset)
        if m.family == "bolinas"
    ]
    if not new_rows:
        return body
    anchored_block = (
        REPRO_GLM_ANCHOR_START
        + "\n"
        + "\n".join(new_rows)
        + "\n"
        + REPRO_GLM_ANCHOR_END
    )
    if match := REPRO_GLM_ANCHORED_RE.search(body):
        return body[: match.start()] + anchored_block + body[match.end() :]
    # First-time migration: locate the existing exp-prefixed gLM block and
    # replace it with the anchored version. Subsequent runs use anchors.
    if match := REPRO_GLM_LEGACY_RE.search(body):
        return body[: match.start()] + anchored_block + "\n" + body[match.end() :]
    raise RuntimeError(
        "could not find gLM block in Reproducibility section "
        "(neither anchored nor legacy `exp`-prefixed rows matched)"
    )


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
    calibrated = score_type_for("gpn_star", "cLLR", dataset)
    raw = score_type_for("gpn_star", "LLR", dataset)
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


def _check_methods_yaml_consistency() -> None:
    """Fail loud if a bolinas entry in dashboard/models.yaml has no
    corresponding model in the live evals_v2 config — catches drift between
    the leaderboard's curated list and the pipeline's source of truth (e.g.
    a config rename without a corresponding models.yaml update)."""
    config_models = _load_evals_v2_config_models()
    missing = [
        m.id
        for ds in DATASETS
        for m in models_for_dataset(ds)
        if m.family == "bolinas" and m.id not in config_models
    ]
    # Dedup while preserving order.
    seen: set[str] = set()
    missing = [i for i in missing if not (i in seen or seen.add(i))]
    if missing:
        raise RuntimeError(
            f"bolinas methods in dashboard/models.yaml not found in "
            f"{EVALS_V2_CONFIG}: {missing}. Either add them to the config "
            f"or remove them from models.yaml."
        )


def main() -> None:
    _check_methods_yaml_consistency()
    parser = argparse.ArgumentParser()
    patch_group = parser.add_mutually_exclusive_group()
    patch_group.add_argument(
        "--patch-issues",
        action="store_true",
        help="After printing, surgically PATCH the bodies of #161/#162/#172.",
    )
    patch_group.add_argument(
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

---
name: agent-research
description: Long-running, iterative investigation tracked in its own GitHub issue — the issue body is the living results doc, comments are the append-only iteration log, and every code reference is a commit-pinned permalink. Use when the user asks to investigate a question over multiple sessions and track the work in a dedicated issue. Common triggers include "let's investigate X and track it in an issue", "open an issue for this and we'll iterate", "dig into Y across sessions", "why is Z slow — track it", "explore this dataset and log findings as we go". Do NOT use for: running a pipeline by itself (even a snakemake/analysis run), single-session debugging or fixes, one-off questions, or regular feature PRs.
---

# agent-research

Inspired by [marin's `agent-research` skill](https://github.com/marin-community/marin/blob/main/.agents/skills/agent-research/SKILL.md), trimmed for this repo (see [§ What this skill is NOT](#what-this-skill-is-not) for what was dropped).

## When to use this skill

Use this skill for long-running, exploratory research where an agent iterates on benchmarks, experiments, and hypotheses over multiple sessions.

The workflow optimizes for:

- **Reproducibility** — every claim links to the exact code that produced it.
- **Clear decision history** — scope changes and course-corrections are visible, not silent.
- **Fast iteration loops** — short append-only comments; heavier body edits only when the headline story shifts.
- **Handoff quality** — a cold reader (human or future agent) can open the issue and catch up.

The question might be EDA-flavored, algorithm-optimization-flavored, or pipeline-investigation-flavored — the skill doesn't care. What matters: findings accumulate over more than one session, and the user has asked to track the work in its own issue.

Do NOT use for:

- Running a pipeline by itself — not a trigger, even if it's a `snakemake/analysis/*` run.
- Single-session debugging or fixes.
- Tasks without an explicit ask to track them.

## Artifacts

Three things. No more.

- A **GitHub issue** labeled `agent-generated` plus one topic label chosen per case (`eda`, `experiment`, `infrastructure`, etc. — reuse existing labels; don't invent a new workflow label).
- *Optionally* a `snakemake/analysis/<name>/` pipeline if the investigation produces one. Its README stays strictly how-to-run.
- Plots and data files attached via the [gh-upload-asset](../gh-upload-asset/SKILL.md) skill.

## Kickoff

1. Draft the issue body using the structure in [§ Issue body structure](#issue-body-structure).
2. Create the issue:

   ```
   gh issue create \
     --title "<short phrasing of the question>" \
     --label agent-generated --label <topic> \
     --body-file <path>
   ```

   No `🤖` prefix on the title — [CLAUDE.md](../../../CLAUDE.md) mandates the prefix on *comments*, and the `agent-generated` label already signals provenance.
3. If a pipeline is needed, scaffold `snakemake/analysis/<name>/` following an existing example (e.g. `snakemake/analysis/sequence_similarity/`) and link it from the issue body.

The repo's `.github/ISSUE_TEMPLATE/experiment.md` is WandB-flavored and does not match this workflow — do not route through it.

## Issue body structure

Order matters: the first screenful is the current state, not the history.

- **Question** — what we're trying to learn, 2–4 sentences, written for a reader who knows genomics and ML broadly but not this thread.
- **Scope** — short bullets of what's in and what's explicitly *out*. Changes here are announced in a `🤖` comment first (see [§ Iteration loop](#iteration-loop)) before being reflected in the body.
- **Approach** — data inputs, methods, link to the pipeline if any.
- **Current findings** — living; edited as results settle. Plots via gist links from `gh-upload-asset`.
- **Open questions / next steps** — edited, not appended.
- **Tracking** — one-liner: *"Description = current state. Comments = append-only iteration log. Pipeline README (if any) = how-to-run only."*

## Iteration loop

For each new run, finding, or course-correction:

- Upload any plots via `gh-upload-asset`.
- Post a **new** `gh issue comment` prefixed with `🤖`. One comment per atomic update. Content is append-only: don't edit a comment to change what it said. Formatting fixes (typos, broken markdown, re-wrapping) are fine.
- **Reference code with permalinks, always.** Every code reference in a comment uses a GitHub permalink pinned to a commit SHA — never `main` or a branch name. Shape:

  ```
  https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<line>
  ```

  Quick ways to produce one (substitute your own `<path>` and `<line>`):

  ```
  # copy a permalink for the current HEAD, a specific line in a file
  gh browse src/bolinas/intervals.py:42 -n -c "$(git rev-parse HEAD)"

  # or construct it yourself
  SHA=$(git rev-parse HEAD)
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
  echo "https://github.com/$REPO/blob/$SHA/src/bolinas/intervals.py#L42"
  ```

  Both forms require the SHA to be pushed to a branch visible on GitHub — otherwise the URL will 404 for readers. If the referenced code isn't yet pushed, push first (to the current branch — *not* `main`) before posting the link.

- If the takeaway changes the headline story, **also** edit the issue body's *Current findings* / *Open questions*. Body edits are retroactive-safe; comments are not.
- **Scope changes get their own comment.** When the thread's direction shifts — adding a question, dropping one, narrowing methodology — post a `🤖` comment that names the shift explicitly ("Scope update: dropping X, adding Y because Z"), *then* update the *Scope* section of the body. Never rewrite *Scope* silently.

## Closure

- Summarize outcomes in the body (*Current findings* becomes final findings; *Open questions* lists what was left).
- Propose close in a final `🤖` comment.
- **Wait for explicit user approval before closing.** Agents don't close issues on their own, per [CLAUDE.md](../../../CLAUDE.md) *Autonomy Boundaries*.

## What this skill is NOT

Deliberately dropped from marin's fuller version of this workflow so the skill stays minimal:

- No research logbook files (`.agents/logbooks/*.md`).
- No Weights & Biases integration.
- No experiment ID prefixes.
- No annotated snapshot tags.
- No dedicated long-lived research branch.

If a thread grows large enough that one of these would help, the user will say so.

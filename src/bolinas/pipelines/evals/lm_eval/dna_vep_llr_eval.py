# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLR-based variant effect prediction (VEP) evaluation task for lm-eval-harness.

Computes log-likelihood ratios (LLR) for DNA sequence variants and evaluates
them with **PairwiseAccuracy ± SE** (matched-pair within-``match_group``
accuracy, ties = 0.5) — the same metric the offline batched VEP path
(``snakemake/analysis/evals_v2/``) reports and the matched-pair leaderboards
#161/#162/#172 live on.

Each row is scored by computing:

    LLR   = log P(alt_completion | context) - log P(ref_completion | context)
    score = llr_transform(LLR)

For datasets that emit two rows per variant (one per strand, ``strand`` column
in ``{"+", "-"}``), the aggregation builds three per-variant scores —
``score_fwd``, ``score_rc``, and ``score_avg = (score_fwd + score_rc) / 2`` —
and reports PairwiseAccuracy on each, keyed as
``{subset}/{fwd|rc|avg}/pairwise_accuracy``. ``score_avg`` matches what the
offline ``snakemake/analysis/evals_v2/`` pipeline computes with
``inference.rc_avg=true`` (#175 conclusion 2); ``fwd`` and ``rc`` are exposed
so the leaderboards can show all three side-by-side. For 1-strand datasets
only the ``avg`` keys are emitted (= the single-strand score). Per-subset,
plus ``_global_`` and ``_macro_avg_`` sentinels via ``compute_pairwise_metrics``.

The lm-eval headline scalar is ``_global_/avg/pairwise_accuracy``.

Ported from ``marin-community/marin@dna-dev``:
``experiments/evals/custom_tasks/dna_vep/dna_vep_llr_eval.py`` (originally
AUPRC; switched to PairwiseAccuracy + per-strand breakdown in #179 for parity
with evals_v2).
"""

from collections import defaultdict
from collections.abc import Callable

import datasets
import pandas as pd
from lm_eval.api.instance import Instance
from lm_eval.api.task import Task

from bolinas.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_pairwise_metrics,
)


# Per-subset PA only reported for subsets with ``n_pairs`` >= this many pairs.
# Matches ``compute_pairwise_metrics``' default ``n_min`` and the leaderboard
# convention (#161/#162/#172): a per-subset cell is only shown when there
# are at least this many matched-pair examples to estimate from. _global_ and
# _macro_avg_ rows are always reported (their ``n_pairs`` has a different
# meaning).
_MIN_PAIRS_PER_SUBSET = 30


# Strand tag → key segment used when storing per-strand PA in the wandb store.
# "+" / "-" → "fwd" / "rc" because slashes in wandb keys group the panel; "+"
# and "-" would render as math operators in Workspace search.
_STRAND_TAGS = {"+": "fwd", "-": "rc"}


class _PairwiseAccuracyAggregation:
    """Aggregation callable that builds per-variant FWD / RC / AVG scores from
    strand-tagged per-row scores, then computes per-subset PairwiseAccuracy ± SE
    via :func:`bolinas.pipelines.evals.metrics.compute_pairwise_metrics`.

    For each variant:
      - if both strand rows are present: emit ``score_fwd``, ``score_rc``, ``score_avg``
      - if only one strand row is present: emit only ``score_avg`` (= the single-strand score)

    All score columns are passed to ``compute_pairwise_metrics`` in a single
    call. The resulting per-subset PA goes into ``self.results_store`` keyed as
    ``{subset}/{strand_tag}/{metric_name}`` (and ``..._se``) where
    ``strand_tag`` ∈ ``{"fwd", "rc", "avg"}``. Per-subset rows with fewer than
    ``_MIN_PAIRS_PER_SUBSET`` matched pairs are dropped (leaderboard
    convention — only stable subset cells get a number); the ``_global_`` and
    ``_macro_avg_`` rows are always emitted. Returns ``_global_/avg/{metric_name}``
    as the lm-eval scalar.
    """

    def __init__(
        self, results_store: dict, metric_name: str, task_name: str | None = None
    ):
        self.results_store = results_store
        self.metric_name = metric_name
        # Used as the wandb-key prefix when pushing per-subset cells to the
        # tracker (so they group under ``lm_eval/<task_name>/...`` like
        # ``log_report_to_tracker`` does for the scalar return value).
        self.task_name = task_name

    def __call__(
        self,
        items: list[tuple[float, float, str | None, tuple, int, str]],
    ) -> float:
        # Group rows per variant. Asserts strand uniqueness within a variant
        # and target/subset/match_group consistency — a silent strand
        # duplication or match_group split would be the kind of bug worth
        # crashing on rather than producing a quietly wrong number.
        by_variant: dict[tuple, dict] = defaultdict(dict)
        for score, target, subset, variant_id, match_group, strand in items:
            v = by_variant[variant_id]
            assert strand in _STRAND_TAGS, (
                f"variant {variant_id} has unknown strand={strand!r}; "
                f"expected one of {sorted(_STRAND_TAGS)}"
            )
            assert strand not in v, (
                f"variant {variant_id} has duplicate strand={strand!r} rows"
            )
            v[strand] = float(score)
            if "_meta" in v:
                m = v["_meta"]
                assert m["target"] == target, (
                    f"variant {variant_id} has inconsistent target: {m['target']} vs {target}"
                )
                assert m["subset"] == subset, (
                    f"variant {variant_id} has inconsistent subset: {m['subset']} vs {subset}"
                )
                assert m["match_group"] == match_group, (
                    f"variant {variant_id} has inconsistent match_group: {m['match_group']} vs {match_group}"
                )
            else:
                v["_meta"] = {
                    "target": target,
                    "subset": subset,
                    "match_group": match_group,
                }

        # Sanity: all variants should carry the same set of strands. A mixed
        # dataset (some 1-strand, some 2-strand variants) would silently make
        # `score_avg` mean different things across rows.
        strand_sets = {
            frozenset(k for k in v if k in _STRAND_TAGS) for v in by_variant.values()
        }
        assert len(strand_sets) == 1, (
            f"variants have heterogeneous strand sets: {sorted(map(sorted, strand_sets))}"
        )
        present_strands = next(iter(strand_sets))
        emit_per_strand = len(present_strands) > 1

        rows: list[dict] = []
        for variant_id, v in by_variant.items():
            meta = v["_meta"]
            row = {
                "label": int(meta["target"]),
                "subset": str(meta["subset"]),
                "match_group": int(meta["match_group"]),
            }
            strand_scores = [v[s] for s in present_strands]
            row["score_avg"] = sum(strand_scores) / len(strand_scores)
            if emit_per_strand:
                for s in present_strands:
                    row[f"score_{_STRAND_TAGS[s]}"] = v[s]
            rows.append(row)

        df = pd.DataFrame(rows)
        score_columns = ["score_avg"]
        if emit_per_strand:
            # Order: fwd, rc, avg — matches how wandb dashboards typically list them.
            score_columns = [
                f"score_{_STRAND_TAGS[s]}" for s in ("+", "-") if s in present_strands
            ] + ["score_avg"]

        metrics = compute_pairwise_metrics(
            dataset=df[["label", "subset", "match_group"]],
            scores=df[score_columns],
            score_columns=score_columns,
            n_min=_MIN_PAIRS_PER_SUBSET,
        )
        # Store every (score_type, subset) row keyed as {subset}/{strand_tag}/{metric_name}.
        # Per-subset rows with fewer than _MIN_PAIRS_PER_SUBSET pairs are dropped
        # to match the leaderboard convention (only _global_ + _macro_avg_ +
        # qualifying per-subset cells are exposed).
        for _, row in metrics.iterrows():
            subset_name = row["subset"]
            n_pairs = int(row["n_pairs"])
            if (
                subset_name not in (GLOBAL_SUBSET, MACRO_AVG_SUBSET)
                and n_pairs < _MIN_PAIRS_PER_SUBSET
            ):
                continue
            strand_tag = row["score_type"].removeprefix("score_")  # fwd / rc / avg
            key = f"{subset_name}/{strand_tag}/{self.metric_name}"
            self.results_store[key] = float(row["value"])
            self.results_store[f"{key}_se"] = float(row["se"])

        # lm-eval only propagates the aggregation's scalar return value into
        # its results JSON (later picked up by levanter's log_report_to_tracker
        # for the wandb summary). Push the per-subset / per-strand cells we
        # computed above to the wandb summary table directly via levanter's
        # tracker — that's the only way they actually surface for the user.
        # Best-effort: skip silently if no tracker is set (e.g. unit tests).
        self._push_per_subset_to_tracker()

        # Headline scalar returned to lm-eval: global PA on the AVG score.
        global_avg = metrics[
            (metrics["subset"] == GLOBAL_SUBSET)
            & (metrics["score_type"] == "score_avg")
        ]
        assert not global_avg.empty, (
            "compute_pairwise_metrics did not emit a _global_ row for score_avg"
        )
        return float(global_avg["value"].iloc[0])

    def _push_per_subset_to_tracker(self) -> None:
        """Push everything in ``self.results_store`` to the levanter tracker as
        logged metrics at ``step=0``, prefixed as
        ``lm_eval/<task_name>/<subset>/<strand>/<metric>`` to match the
        convention ``levanter.eval_harness.log_report_to_tracker`` uses for
        the scalar return value.

        Use ``log(step=0)`` (not ``log_summary``): the wandb backend then
        writes both run history (so the cells show up as workspace charts)
        and the summary panel (auto-filled from the latest logged value).
        ``log_summary`` only populated the Overview-page summary table.

        Best-effort: silently no-ops if levanter isn't importable or no tracker
        is set (unit tests, ad-hoc scripts).
        """
        try:
            import levanter.tracker
        except ImportError:
            return
        prefix = "lm_eval"
        if self.task_name:
            prefix = f"{prefix}/{self.task_name}"
        payload = {
            f"{prefix}/{key}": value for key, value in self.results_store.items()
        }
        try:
            levanter.tracker.log(payload, step=0)
        except Exception:
            # NoopTracker / no current tracker / serialization issue: don't
            # take down the entire eval because of a logging side-channel.
            pass


METRIC_REGISTRY: dict[str, dict] = {
    "pairwise_accuracy": {
        "aggregation_cls": _PairwiseAccuracyAggregation,
        "higher_is_better": True,
    },
}


LLR_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": lambda x: x,
    "negate": lambda x: -x,
    "abs": abs,
}


class DnaVepLlrEvalTask(Task):
    """LLR eval task for variant effect prediction with per-strand + AVG aggregation.

    Parameterized by dataset and metrics via YAML config.

    Dataset rows must have ``[chrom, pos, ref, alt, context, ref_completion,
    alt_completion, target, match_group]``. Optional: ``subset`` (str) — metrics
    computed per distinct value plus ``_global_`` and ``_macro_avg_``.
    Optional: ``strand`` (str, ``"+"`` or ``"-"``) — when present the same variant
    appears once per strand (see
    :func:`bolinas.pipelines.evals.materialize.materialize_sequences`).

    Aggregation: per-row scores are grouped by ``(chrom, pos, ref, alt)``. For a
    2-strand dataset the metric is computed three times — on per-variant
    ``score_fwd``, ``score_rc``, and ``score_avg = (score_fwd + score_rc) / 2``
    — and stored under keys ``{subset}/fwd/{metric}``, ``{subset}/rc/{metric}``,
    ``{subset}/avg/{metric}`` (plus ``..._se``). For a 1-strand dataset only
    the ``avg`` keys are emitted (= the single-strand score). The lm-eval
    headline scalar is ``_global_/avg/{metric}``.

    YAML config fields:
        dataset_path: HuggingFace dataset path
        dataset_name: HuggingFace dataset config name (optional)
        test_split: dataset split to evaluate on
        metrics: list of metric names from ``METRIC_REGISTRY`` (just
            ``pairwise_accuracy`` is supported as of #179).
        llr_transform: identity | negate | abs (default: identity)
    """

    # Bumped in #179: switched from per-row AUPRC to per-variant PairwiseAccuracy
    # (3-tuple → 5-tuple in process_results, metric registry entirely replaced).
    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(
        self, data_dir=None, cache_dir=None, download_mode=None, config=None
    ) -> None:
        # Task.__init__ calls self.download() BEFORE setting self._config,
        # and wraps the config dict via TaskConfig({**config}) which doesn't
        # correctly populate fields (passes the dict as the first positional
        # arg). We extract everything we need from the raw config dict here.
        cfg = config or {}
        self._task_name = cfg.get("task")
        self.DATASET_PATH = cfg.get("dataset_path") or self.DATASET_PATH
        self.DATASET_NAME = cfg.get("dataset_name") or self.DATASET_NAME
        self._metrics = cfg.get("metrics") or ["pairwise_accuracy"]
        self._test_split = cfg.get("test_split") or "test"
        unknown = [m for m in self._metrics if m not in METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}. Must be one of {list(METRIC_REGISTRY)}"
            )

        transform_name = cfg.get("llr_transform") or "identity"
        if transform_name not in LLR_TRANSFORMS:
            raise ValueError(
                f"Unknown llr_transform: {transform_name}. Must be one of {list(LLR_TRANSFORMS.keys())}"
            )
        self._llr_transform = LLR_TRANSFORMS[transform_name]
        self._subset_results: dict[str, float] = {}

        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config,
        )

    @property
    def task_name(self) -> str:
        """Required by lm-eval's get_subtask_list (only defined on ConfigurableTask by default)."""
        return self._task_name

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset[self._test_split]

    def doc_to_text(self, doc) -> str:
        return doc["context"]

    def doc_to_target(self, doc):
        raise NotImplementedError(
            "DnaVepLlrEvalTask does not use doc_to_target. "
            "It overrides construct_requests and should be used with num_fewshot=0."
        )

    def construct_requests(self, doc, ctx, **kwargs):
        # Only pass fields that Instance accepts; drop chat template args
        # passed by build_all_requests.
        metadata = kwargs.get("metadata", (None, None, None))
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["ref_completion"]),
                idx=0,
                metadata=metadata,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["alt_completion"]),
                idx=1,
                metadata=metadata,
            ),
        ]

    def process_results(self, doc, results):
        log_prob_ref = results[0][0]
        log_prob_alt = results[1][0]
        llr = log_prob_alt - log_prob_ref
        score = self._llr_transform(llr)
        target = doc["target"]
        subset = doc.get("subset")
        # Defensive casts because HF can return numpy scalars in dataset rows;
        # the variant_id tuple needs to be hashable for per-variant collapse.
        variant_id = (
            str(doc["chrom"]),
            int(doc["pos"]),
            str(doc["ref"]),
            str(doc["alt"]),
        )
        match_group = int(doc["match_group"])
        # Default to "+" for legacy 1-row-per-variant datasets that don't
        # carry a strand column. The aggregation will still emit only score_avg
        # in that case (see _PairwiseAccuracyAggregation.__call__).
        strand = str(doc.get("strand", "+"))
        return {
            metric: (score, target, subset, variant_id, match_group, strand)
            for metric in self._metrics
        }

    def aggregation(self):
        return {
            metric: METRIC_REGISTRY[metric]["aggregation_cls"](
                results_store=self._subset_results,
                metric_name=metric,
                task_name=self._task_name,
            )
            for metric in self._metrics
        }

    def higher_is_better(self):
        return {
            metric: METRIC_REGISTRY[metric]["higher_is_better"]
            for metric in self._metrics
        }

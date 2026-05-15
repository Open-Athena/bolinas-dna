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

When the dataset emits two rows per variant (one per strand, ``strand`` column
in ``{"+", "-"}``), the per-strand scores are averaged per
``(chrom, pos, ref, alt)`` before the metric is computed — matches
``snakemake/analysis/evals_v2/`` ``inference.rc_avg=true`` (FWD+RC strand
averaging, #175 conclusion 2). For datasets with one row per variant the
averaging step is a no-op.

Per-subset metrics are reported as ``{subset}/pairwise_accuracy`` plus
``_global_/pairwise_accuracy`` and ``_macro_avg_/pairwise_accuracy`` sentinel
rows from ``compute_pairwise_metrics``. The headline scalar returned to
lm-eval is the global PA.

Ported from ``marin-community/marin@dna-dev``:
``experiments/evals/custom_tasks/dna_vep/dna_vep_llr_eval.py`` (originally
AUPRC; switched to PairwiseAccuracy in #179 for parity with evals_v2).
"""

from collections import defaultdict
from collections.abc import Callable

import datasets
import pandas as pd
from lm_eval.api.instance import Instance
from lm_eval.api.task import Task

from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, compute_pairwise_metrics


class _PairwiseAccuracyAggregation:
    """Aggregation callable that averages per-variant scores across rows
    (FWD/RC strand collapse) and computes per-subset PairwiseAccuracy ± SE
    via :func:`bolinas.pipelines.evals.metrics.compute_pairwise_metrics`.

    Stores every row of the metrics DataFrame in ``self.results_store``
    keyed as ``{subset}/pairwise_accuracy`` and
    ``{subset}/pairwise_accuracy_se`` for wandb grouping. Returns the
    ``_global_/pairwise_accuracy`` value as the lm-eval scalar.
    """

    def __init__(self, results_store: dict, metric_name: str):
        self.results_store = results_store
        self.metric_name = metric_name

    def __call__(
        self,
        items: list[tuple[float, float, str | None, tuple, int]],
    ) -> float:
        # Collapse per variant_id (averaging scores across rows = FWD/RC strands).
        # Asserts target/subset/match_group are consistent within a variant — a
        # silent doubling or a match_group split would be the kind of bug worth
        # crashing on rather than producing a quietly wrong number.
        by_variant: dict[tuple, list[tuple[float, float, str | None, int]]] = (
            defaultdict(list)
        )
        for score, target, subset, variant_id, match_group in items:
            by_variant[variant_id].append((score, target, subset, match_group))

        rows: list[dict] = []
        for variant_id, group in by_variant.items():
            scores = [g[0] for g in group]
            targets = {g[1] for g in group}
            subsets = {g[2] for g in group}
            match_groups = {g[3] for g in group}
            assert len(targets) == 1, (
                f"variant {variant_id} has inconsistent target across rows: {targets}"
            )
            assert len(subsets) == 1, (
                f"variant {variant_id} has inconsistent subset across rows: {subsets}"
            )
            assert len(match_groups) == 1, (
                f"variant {variant_id} has inconsistent match_group across rows: {match_groups}"
            )
            rows.append(
                {
                    "label": int(targets.pop()),
                    "score": float(sum(scores) / len(scores)),
                    "subset": str(next(iter(subsets))),
                    "match_group": int(match_groups.pop()),
                }
            )

        df = pd.DataFrame(rows)
        metrics = compute_pairwise_metrics(
            dataset=df[["label", "subset", "match_group"]],
            scores=df[["score"]],
            score_columns=["score"],
        )
        for _, row in metrics.iterrows():
            self.results_store[f"{row['subset']}/{self.metric_name}"] = float(
                row["value"]
            )
            self.results_store[f"{row['subset']}/{self.metric_name}_se"] = float(
                row["se"]
            )

        global_row = metrics[metrics["subset"] == GLOBAL_SUBSET]
        assert not global_row.empty, (
            "compute_pairwise_metrics did not emit a _global_ row"
        )
        return float(global_row["value"].iloc[0])


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
    """LLR eval task for variant effect prediction with per-variant aggregation.

    Parameterized by dataset and metrics via YAML config.

    Dataset rows must have ``[chrom, pos, ref, alt, context, ref_completion,
    alt_completion, target, match_group]``. Optional: ``subset`` (str) — metrics
    computed per distinct value plus ``_global_`` and ``_macro_avg_``.

    When the dataset has multiple rows per variant (e.g. two rows per variant
    tagged by ``strand`` in ``{"+", "-"}`` — see
    :func:`bolinas.pipelines.evals.materialize.materialize_sequences`), per-row
    scores are averaged per ``(chrom, pos, ref, alt)`` before the
    PairwiseAccuracy is computed. For one-row-per-variant datasets the
    averaging step is a no-op.

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
        return {
            metric: (score, target, subset, variant_id, match_group)
            for metric in self._metrics
        }

    def aggregation(self):
        return {
            metric: METRIC_REGISTRY[metric]["aggregation_cls"](
                results_store=self._subset_results,
                metric_name=metric,
            )
            for metric in self._metrics
        }

    def higher_is_better(self):
        return {
            metric: METRIC_REGISTRY[metric]["higher_is_better"]
            for metric in self._metrics
        }

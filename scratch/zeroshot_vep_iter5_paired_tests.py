"""Iter-5 follow-up to #175: paired McNemar-style sign tests on the FWD+RC AVG
exp166-p1B scores.

Two question families:

Q1) Is any of the 3 "approaches" — LLR (`minus_llr`/`abs_llr`), `embed_l2_flat_last`,
    or ensembling — better than each other? For ensembling we pick the best ensemble
    per (dataset, Global) — this is the question "does ensembling actually help
    relative to either base score". 3 comparisons per dataset:
      - LLR vs L2
      - LLR vs best-ensemble (by Global PA)
      - L2 vs best-ensemble

Q2) Among the 7 ensembles, does any beat the standard `mean_rank` baseline? 6
    comparisons per dataset (mean_rank vs each of the other 6).

The test: ``paired_score_comparison`` from ``bolinas.evals.metrics``. For each
``match_group`` (one pos + one neg) both scores produce an ordering outcome
∈ {0, 0.5, 1}. The per-pair advantage is ``out_A − out_B`` ∈ {-1, -0.5, 0, +0.5, +1}.
Under H0, advantages are symmetric around 0 → wins ~ Binom(n_discordant, 0.5).
Two-sided alternative; BH-FDR adjustment within (question, dataset).

Operates at the **Global** level (PA over all match-pairs concatenated). Macro
Avg-level paired tests are not naturally defined (too few subsets for complex
where k=2); we only report Global p-values here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bolinas.evals.metrics import paired_score_comparison
from bolinas.zeroshot_vep.scores import SCORE_NAMES, apply_score_directions


DATASETS = ["mendelian_traits", "complex_traits", "eqtl"]
LLR_BY_DATASET = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
    "eqtl": "abs_llr",
}
EMBED = "embed_l2_flat_last"

KEY = ["chrom", "pos", "ref", "alt"]
ENSEMBLES = ["mean_rank", "min_rank", "max_rank", "geomean_rank",
             "harmonic_rank", "rrf_k60", "zscore_mean"]
# Ensembles excluding the mean_rank baseline (for Q2)
ENSEMBLES_VS_MEAN = [c for c in ENSEMBLES if c != "mean_rank"]


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    v = p[mask]
    order = np.argsort(v)
    n = len(v)
    ranks = np.arange(1, n + 1)
    q_sorted = v[order] * n / ranks
    q_monotonic = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_clipped = np.minimum(q_monotonic, 1.0)
    out_mask = np.empty(n, dtype=np.float64)
    out_mask[order] = q_clipped
    out[mask] = out_mask
    return out


def _rc_path(ds: str) -> str:
    if ds == "mendelian_traits":
        return f"scratch/iter4/iter4_rc_exp166-p1B__win255__{ds}.parquet"
    return f"scratch/iter5/iter5_rc_exp166-p1B__win255__{ds}.parquet"


def _fwd_path(ds: str) -> str:
    return f"scratch/iter4/iter4_fwd_exp166-p1B__win255__{ds}.parquet"


def _build_score_table(ds: str) -> pd.DataFrame:
    """Reproduces the iter-5 ensembles end-to-end on FWD+RC AVG signed scores."""
    fwd = pd.read_parquet(_fwd_path(ds))
    rc = pd.read_parquet(_rc_path(ds))
    rc_renamed = rc[KEY + list(SCORE_NAMES)].rename(
        columns={c: f"{c}__rc" for c in SCORE_NAMES}
    )
    df = fwd.merge(rc_renamed, on=KEY, how="inner")
    assert len(df) == len(fwd) == len(rc), f"{ds}: FWD/RC row count mismatch"

    signed_fwd = apply_score_directions(df[SCORE_NAMES])
    rc_raw = df[[f"{c}__rc" for c in SCORE_NAMES]].rename(
        columns=lambda c: c.replace("__rc", "")
    )
    signed_rc = apply_score_directions(rc_raw)
    signed_avg = (signed_fwd + signed_rc) / 2.0

    llr_name = LLR_BY_DATASET[ds]
    out = pd.DataFrame({
        "subset": df["subset"].values,
        "label": df["label"].values,
        "match_group": df["match_group"].values,
        llr_name: signed_avg[llr_name].values,
        EMBED: signed_avg[EMBED].values,
    })

    g = out.groupby("subset", sort=False)
    r_llr = g[llr_name].rank(method="average")
    r_emb = g[EMBED].rank(method="average")
    out["mean_rank"] = (r_llr + r_emb) / 2.0
    out["min_rank"] = np.minimum(r_llr.values, r_emb.values)
    out["max_rank"] = np.maximum(r_llr.values, r_emb.values)
    out["geomean_rank"] = np.sqrt(r_llr.values * r_emb.values)
    out["harmonic_rank"] = 2.0 / (1.0 / r_llr.values + 1.0 / r_emb.values)
    n_per_subset = g[llr_name].transform("count").values
    rft_llr = n_per_subset + 1.0 - r_llr.values
    rft_emb = n_per_subset + 1.0 - r_emb.values
    out["rrf_k60"] = 1.0 / (60.0 + rft_llr) + 1.0 / (60.0 + rft_emb)

    def _z(col):
        mean = g[col].transform("mean")
        std = g[col].transform("std").replace(0, 1.0)
        return ((out[col] - mean) / std).values
    out["zscore_mean"] = (_z(llr_name) + _z(EMBED)) / 2.0

    return out


def _paired(scored: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Run paired_score_comparison at the Global level (all match-pairs pooled).

    Returns dict with: value (frac pairs A wins), n_pairs, n_a_wins, n_b_wins,
    n_concordant, n_half, p_value (two-sided).
    """
    return paired_score_comparison(
        label=scored["label"],
        score_a=scored[col_a],
        score_b=scored[col_b],
        match_group=scored["match_group"],
        alternative="two-sided",
    )


def _global_pa(scored: pd.DataFrame, col: str) -> float:
    from bolinas.evals.metrics import pairwise_accuracy
    return pairwise_accuracy(
        scored["label"], scored[col], scored["match_group"], alternative="greater"
    )["value"]


def main() -> int:
    out_dir = Path("scratch/iter5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Q1: LLR vs L2 vs best ensemble -----
    q1_rows = []
    best_ens_per_dataset: dict[str, str] = {}
    for ds in DATASETS:
        scored = _build_score_table(ds)
        llr_name = LLR_BY_DATASET[ds]
        # Best ensemble by Global PA
        pas = {c: _global_pa(scored, c) for c in ENSEMBLES}
        best_ens = max(pas.items(), key=lambda kv: kv[1])[0]
        best_ens_per_dataset[ds] = best_ens

        # 3 pairwise comparisons. Make A be the "candidate", B the "baseline" so
        # value > 0.5 means candidate wins.
        comparisons = [
            ("L2", EMBED, "LLR", llr_name),
            (f"best_ens={best_ens}", best_ens, "LLR", llr_name),
            (f"best_ens={best_ens}", best_ens, "L2", EMBED),
        ]
        for label_a, col_a, label_b, col_b in comparisons:
            r = _paired(scored, col_a, col_b)
            q1_rows.append({
                "dataset": ds, "A": label_a, "B": label_b,
                "PA_A": _global_pa(scored, col_a),
                "PA_B": _global_pa(scored, col_b),
                "frac_A_wins": r["value"],
                "n_pairs": r["n_pairs"],
                "n_a_wins": r["n_a_wins"], "n_b_wins": r["n_b_wins"],
                "n_concordant": r["n_concordant"], "n_half": r["n_half"],
                "p_value": r["p_value"],
            })

    q1 = pd.DataFrame(q1_rows)
    # BH within dataset (3 tests each)
    q1["q_value"] = np.nan
    for ds in DATASETS:
        idx = q1["dataset"] == ds
        q1.loc[idx, "q_value"] = _bh_adjust(q1.loc[idx, "p_value"].values)
    q1.to_parquet(out_dir / "iter5_q1_paired_tests.parquet", index=False)
    print("\n=== Q1: 3-approach paired tests (Global level) ===")
    print(q1.to_string(index=False))

    # ----- Q2: each ensemble vs mean_rank -----
    q2_rows = []
    for ds in DATASETS:
        scored = _build_score_table(ds)
        for ens in ENSEMBLES_VS_MEAN:
            r = _paired(scored, ens, "mean_rank")
            q2_rows.append({
                "dataset": ds, "candidate": ens, "baseline": "mean_rank",
                "PA_candidate": _global_pa(scored, ens),
                "PA_baseline": _global_pa(scored, "mean_rank"),
                "frac_candidate_wins": r["value"],
                "n_pairs": r["n_pairs"],
                "n_cand_wins": r["n_a_wins"], "n_base_wins": r["n_b_wins"],
                "n_concordant": r["n_concordant"], "n_half": r["n_half"],
                "p_value": r["p_value"],
            })
    q2 = pd.DataFrame(q2_rows)
    q2["q_value"] = np.nan
    for ds in DATASETS:
        idx = q2["dataset"] == ds
        q2.loc[idx, "q_value"] = _bh_adjust(q2.loc[idx, "p_value"].values)
    q2.to_parquet(out_dir / "iter5_q2_paired_tests.parquet", index=False)
    print("\n=== Q2: ensembles vs mean_rank paired tests (Global level) ===")
    print(q2.to_string(index=False))

    # ----- Render comments -----
    _write_q1_comment(q1, best_ens_per_dataset, out_dir)
    _write_q2_comment(q2, out_dir)

    return 0


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _write_q1_comment(q1: pd.DataFrame, best_ens: dict, out_dir: Path) -> None:
    lines = [
        "🤖 **Q1: paired sign-test — is any of {LLR, L2, ensemble} better than the others?**",
        "",
        "Test: `paired_score_comparison` (McNemar-style sign test, two-sided) at the Global level "
        "(all match-pairs pooled per dataset). `frac_A_wins` = fraction of pairs where A correctly orders "
        "(pos>neg) but B does not, with halves counted at 0.5 (note: A only \"wins\" a pair when "
        "it disagrees with B, so concordant pairs aren't counted). BH-adjusted q within each dataset (3 tests).",
        "",
        "For \"ensemble\" we pick the best ensemble per dataset by Global PairwiseAccuracy:",
        "",
    ]
    for ds in DATASETS:
        lines.append(f"- **{ds}**: best ensemble = `{best_ens[ds]}`")
    lines.append("")

    for ds in DATASETS:
        sub = q1[q1["dataset"] == ds].copy()
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| comparison (A vs B) | PA(A) | PA(B) | frac A wins | n_pairs | n_A>B | n_B>A | n_tied | p (two-sided) | q (BH) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            sig = " ★" if r["q_value"] < 0.05 else ""
            tied = int(r["n_concordant"]) + int(r["n_half"])
            lines.append(
                f"| {r['A']} vs {r['B']} | {r['PA_A']:.4f} | {r['PA_B']:.4f} | "
                f"{r['frac_A_wins']:.4f} | {int(r['n_pairs'])} | "
                f"{int(r['n_a_wins'])} | {int(r['n_b_wins'])} | {tied} | "
                f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
            )
        lines.append("")

    lines.append("★ = q < 0.05.")
    lines.append("")
    lines.append("**Reading**: `PA(A) > PA(B)` alone is descriptive; the paired test is the rigorous check that "
                 "the per-pair advantage is non-random. Concordant pairs (both A and B order correctly, or both wrong) "
                 "carry no signal for the test.")

    text = "\n".join(lines)
    (out_dir / "iter5_q1_comment.md").write_text(text)
    print(f"\n[iter5] wrote {out_dir / 'iter5_q1_comment.md'}")


def _write_q2_comment(q2: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "🤖 **Q2: among the 7 ensembles, does any beat the standard `mean_rank` baseline?**",
        "",
        "Test: `paired_score_comparison` (McNemar-style, two-sided) at the Global level. "
        "BH-adjusted q within each dataset (6 tests per dataset). Δ-PA = PA(candidate) − PA(mean_rank). "
        "`frac cand wins` is the fraction of *discordant* pairs where the candidate orders correctly "
        "and mean_rank does not (halves = 0.5).",
        "",
    ]
    for ds in DATASETS:
        sub = q2[q2["dataset"] == ds].copy().sort_values("p_value")
        pa_baseline = sub["PA_baseline"].iloc[0]
        lines.append(f"### {ds}  (mean_rank Global PA = {pa_baseline:.4f})")
        lines.append("")
        lines.append("| candidate | PA(cand) | Δ vs mean_rank | frac cand wins | n_pairs | n_cand>base | n_base>cand | n_tied | p (two-sided) | q (BH) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            delta = r["PA_candidate"] - r["PA_baseline"]
            sig = " ★" if r["q_value"] < 0.05 else ""
            tied = int(r["n_concordant"]) + int(r["n_half"])
            lines.append(
                f"| `{r['candidate']}` | {r['PA_candidate']:.4f} | {delta:+.4f} | "
                f"{r['frac_candidate_wins']:.4f} | {int(r['n_pairs'])} | "
                f"{int(r['n_cand_wins'])} | {int(r['n_base_wins'])} | {tied} | "
                f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
            )
        lines.append("")
    lines.append("★ = q < 0.05.")

    text = "\n".join(lines)
    (out_dir / "iter5_q2_comment.md").write_text(text)
    print(f"[iter5] wrote {out_dir / 'iter5_q2_comment.md'}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Iter 12 — sweep bin schemes for the simplified iter-11 design.

Goal: confirm retention/leakage are not over-fit to one specific bin schema.

iter-11 design recap:
- tss_dist bins applied ONLY to tss_proximal (else 'NA').
- exon_dist bins applied ONLY to splicing (else 'NA').
- Continuous matching on tss_dist + exon_dist universally.
- OOR fallback for any in-subset value > the cap.

Sweep ranges: bin counts 2..7 for both features. Edges chosen to be roughly
log-spaced within the cap range.
"""
import polars as pl

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

# Each scheme caps at the proximal threshold (tss=1000, exon=30); no outer bin.
TSS_BIN_SCHEMES = {
    "2 bins":          [0, 200, 1000],
    "3 bins":          [0, 100, 500, 1000],
    "4 bins (iter11)": [0, 50, 200, 500, 1000],
    "5 bins":          [0, 25, 100, 300, 700, 1000],
    "6 bins":          [0, 25, 50, 100, 300, 700, 1000],
    "7 bins":          [0, 25, 50, 100, 200, 400, 700, 1000],
}
EXON_BIN_SCHEMES = {
    "2 bins":          [0, 5, 30],
    "3 bins (iter11)": [0, 5, 20, 30],
    "4 bins":          [0, 2, 5, 20, 30],
    "5 bins":          [0, 2, 5, 10, 20, 30],
    "6 bins":          [0, 2, 5, 10, 15, 20, 30],
}


def add_bin_label(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
    n = len(edges) - 1
    expr = pl.lit("OOR")
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        cond = (
            (pl.col(feature) >= lo) & (pl.col(feature) < hi)
            if i < n - 1
            else ((pl.col(feature) >= lo) & (pl.col(feature) <= hi))
        )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return V.with_columns(**{col: expr})


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


def run(tss_edges: list[float], exon_edges: list[float], V_base: pl.DataFrame) -> dict:
    V = add_bin_label(V_base, "tss_dist", tss_edges, "_tss_full")
    V = add_bin_label(V, "exon_dist", exon_edges, "_exon_full")
    V = V.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(pl.col("_tss_full"))
        .otherwise(pl.lit("NA"))
        .alias("tss_dist_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(pl.col("_exon_full"))
        .otherwise(pl.lit("NA"))
        .alias("exon_dist_bin"),
    )
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        ["tss_dist", "exon_dist"],
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
         "tss_dist_bin", "exon_dist_bin"],
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    out = {"total": pos.height}
    for sub_name in ["tss_proximal", "splicing"]:
        sub = V_m.filter(pl.col("consequence_group") == sub_name)
        n = sub.filter(pl.col("label")).height
        out[f"{sub_name}_n"] = n
        if n > 0:
            for f in ["tss_dist", "exon_dist"]:
                out[f"{sub_name}_{f}_pacc"] = pairwise_acc(sub, f)
    return out


print("Loading dataset_all...", flush=True)
V_base = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)


print("\n========== TSS bin scheme sweep (exon = iter-11 default) ==========", flush=True)
print(f"  {'scheme':<20s}  {'total':>6s}  {'tss_n':>6s}  {'tss_dist_pacc':>14s}", flush=True)
default_exon = EXON_BIN_SCHEMES["3 bins (iter11)"]
for name, tss_edges in TSS_BIN_SCHEMES.items():
    r = run(tss_edges, default_exon, V_base)
    print(f"  {name:<20s}  {r['total']:>6d}  {r['tss_proximal_n']:>6d}  "
          f"{r.get('tss_proximal_tss_dist_pacc', float('nan')):>14.3f}", flush=True)

print("\n========== exon bin scheme sweep (tss = iter-11 default) ==========", flush=True)
print(f"  {'scheme':<20s}  {'total':>6s}  {'splicing_n':>10s}  {'splicing_exon_dist_pacc':>22s}", flush=True)
default_tss = TSS_BIN_SCHEMES["4 bins (iter11)"]
for name, exon_edges in EXON_BIN_SCHEMES.items():
    r = run(default_tss, exon_edges, V_base)
    print(f"  {name:<20s}  {r['total']:>6d}  {r['splicing_n']:>10d}  "
          f"{r.get('splicing_exon_dist_pacc', float('nan')):>22.3f}", flush=True)

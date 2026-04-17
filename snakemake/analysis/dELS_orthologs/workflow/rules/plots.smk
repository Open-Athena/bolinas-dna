"""Plots over the class × conservation-bin recall tables.

Two plots per (aligner, track):
- `recall_heatmap.png` — classes on rows, conservation bins on columns,
  cells annotated with recall@1.
- `recall_curves.png` — one line per class, x = conservation bin midpoint,
  y = recall@1; 'Overall' drawn in bold.
"""


rule plot_recall_heatmap:
    input:
        "results/eval/{aligner}/recall_by_class_and_conservation_fixed/{track}.parquet",
    output:
        "results/plots/{aligner}/{track}_recall_heatmap.png",
    run:
        df = pl.read_parquet(input[0]).filter(pl.col("k") == 1).to_pandas()

        pivot = df.pivot_table(index="cre_class", columns="bin_idx", values="recall")
        # Sort classes by aggregate recall (reuse n_queries-weighted mean)
        class_n = df.groupby("cre_class")["n_queries"].sum()
        class_hits = df.groupby("cre_class")["n_hits"].sum()
        class_recall = (class_hits / class_n).sort_values(ascending=False)
        # Move 'Overall' to the top for readability
        ordered = ["Overall"] + [c for c in class_recall.index if c != "Overall"]
        pivot = pivot.reindex(ordered)

        bin_labels = (
            df.drop_duplicates("bin_idx")
            .sort_values("bin_idx")[["bin_idx", "bin"]]
            .set_index("bin_idx")["bin"]
        )
        pivot.columns = [bin_labels[b] for b in pivot.columns]

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "recall@1"},
            ax=ax,
        )
        ax.set_xlabel(f"% bases with {wildcards.track} above threshold")
        ax.set_ylabel("hg38 cCRE class")
        ax.set_title(
            f"Recall@1 by class × conservation bin — {wildcards.aligner} / {wildcards.track}"
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(output[0])
        plt.close(fig)


rule plot_recall_curves:
    input:
        "results/eval/{aligner}/recall_by_class_and_conservation_fixed/{track}.parquet",
    output:
        "results/plots/{aligner}/{track}_recall_curves.png",
    run:
        df = pl.read_parquet(input[0]).filter(pl.col("k") == 1).to_pandas()

        # Bin midpoints for the x axis — parse from bin label "[a, b)"
        bin_mid = (
            df.drop_duplicates("bin_idx")
            .sort_values("bin_idx")[["bin_idx", "bin"]]
            .assign(
                mid=lambda r: r["bin"].str.extract(r"\[([\d.]+),")[0].astype(float)
                + 0.05  # 10% wide bins; midpoint is lo + 0.05
            )
            .set_index("bin_idx")["mid"]
        )

        # Sort classes by aggregate recall (descending); 'Overall' drawn bold last
        class_n = df.groupby("cre_class")["n_queries"].sum()
        class_hits = df.groupby("cre_class")["n_hits"].sum()
        class_recall_agg = (class_hits / class_n).sort_values(ascending=False)
        classes_ordered = [c for c in class_recall_agg.index if c != "Overall"] + [
            "Overall"
        ]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        palette = sns.color_palette("tab10", n_colors=len(classes_ordered) - 1)
        for i, cls in enumerate(classes_ordered):
            sub = df[df["cre_class"] == cls].sort_values("bin_idx")
            xs = [bin_mid[b] for b in sub["bin_idx"]]
            ys = sub["recall"].to_list()
            if cls == "Overall":
                ax.plot(xs, ys, color="black", lw=2.5, label=cls, zorder=5)
            else:
                ax.plot(
                    xs,
                    ys,
                    color=palette[i],
                    lw=1.4,
                    marker="o",
                    markersize=4,
                    label=cls,
                    alpha=0.85,
                )

        ax.set_xlabel(f"% bases with {wildcards.track} above threshold")
        ax.set_ylabel("recall@1")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.set_title(
            f"Recall@1 vs. conservation, by cCRE class — {wildcards.aligner} / {wildcards.track}"
        )
        fig.tight_layout()
        fig.savefig(output[0])
        plt.close(fig)

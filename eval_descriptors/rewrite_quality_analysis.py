import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd  # type:ignore
import seaborn as sns  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np


def read_similarity_scores(data_version, return_all=False):
    run_ids = [
        "final_zero_vocab",
        "final_50_vocab",
        "final_100_vocab",
        "final_200_vocab",
        "final_300_vocab",
        "final_500_vocab",
    ]
    rewrite_scores = defaultdict(list)

    for run in run_ids:
        if data_version == "final":
            path = Path("..") / "results" / run / f"descriptors_{run}_final.jsonl"
            with path.open("r") as f:
                for line in f:
                    line = json.loads(line, strict=False)
                    rewrite_scores[run].append((line["similarity"][0]))
        elif data_version == "og":
            path = Path("..") / "results" / run / f"descriptors_{run}.jsonl"
            with path.open("r") as f:
                for line in f:
                    line = json.loads(line, strict=False)
                    if return_all:
                        rewrite_scores[run].append(line["similarity"])
                    else:
                        rewrite_scores[run].append(max(line["similarity"]))
    return rewrite_scores


def calculate_avg_rewrite_scores():
    """Calculate the average rewrite quality in the 'final' data versions."""

    rewrite_scores = read_similarity_scores("final")

    save_path = Path("..") / "results" / "evaluations" / "rewrite_score_averages.txt"
    with save_path.open("w") as f:
        for k, v in rewrite_scores.items():
            avg = sum(v) / len(v)
            f.write(f"{k}: {avg}\n")


def basic_stats(og_scores, final_scores):
    # Calculate average scores
    og_avg = round(np.mean(og_scores), 3)
    final_avg = round(np.mean(final_scores), 3)

    # Calculate standard deviation
    og_std = round(np.std(og_scores), 3)
    final_std = round(np.std(final_scores), 3)

    # Calculate max and min
    og_min = round(min(og_scores), 3)
    og_max = round(max(og_scores), 3)
    final_min = round(min(final_scores), 3)
    final_max = round(max(final_scores), 3)

    # Save basic stats
    save_path = (
        Path("..") / "results" / "evaluations" / "og_vs_synonym_rewrite_diffs.txt"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        f.write("similarity score stats between original descriptors and synonyms.\n")
        f.write("Calculated using the zero vocab setting.\n\n")
        f.write(
            f"Original descriptors:\nMean: {og_avg}\n"
            f"Standard deviation: {og_std}\n"
            f"Minimum value: {og_min}\n"
            f"Maximum vaue: {og_max}\n\n"
        )
        f.write(
            f"Synonym descriptors:\nMean: {final_avg}\n"
            f"Standard deviation: {final_std}\n"
            f"Minimum value: {final_min}\n"
            f"Maximum vaue: {final_max}\n"
        )


def plot_score_distribution(og_scores, final_scores):
    bins = np.round(np.arange(start=0, stop=1 + 0.05, step=0.05), 2)
    # Create a DataFrame in long format
    df_og = pd.DataFrame({"value": og_scores, "dataset": "Original descriptors"})
    df_final = pd.DataFrame({"value": final_scores, "dataset": "Synonym descriptors"})
    df = pd.concat([df_og, df_final])

    # Bin the data
    df["bin"] = pd.cut(df["value"], bins=bins, include_lowest=True)

    # Count the number of values per bin per dataset
    bin_counts = df.groupby(["dataset", "bin"]).size().reset_index(name="count")

    # To make bin centers for plotting
    bin_counts["bin_label"] = bin_counts["bin"].apply(
        lambda x: f"{x.left:.1f}–{x.right:.1f}"
    )

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="talk")
    sns.barplot(
        data=bin_counts,
        x="bin_label",
        y="count",
        hue="dataset",
        width=0.8,
        palette="viridis",
    )

    plt.xlabel("Similarity score range")
    plt.ylabel("Count")
    plt.xticks(rotation=45, fontsize=10)

    # Save figure
    fig_path = (
        Path("..") / "figures" / "rewrite_score_distribution_in_zero_vocab_setting.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)


def plot_score_differences(og_scores, final_scores):
    # Calculate differences
    diffs = [final - og for og, final in zip(og_scores, final_scores)]
    diffs_sorted = sorted(diffs)

    # Create bins and histogram
    min_diff, max_diff = min(diffs_sorted), max(diffs_sorted)
    bins = np.round(np.arange(min_diff, max_diff + 0.1, 0.1), 1)  # Create bins of 0.1
    bin_counts, bin_edges = np.histogram(diffs_sorted, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers = np.round(bin_centers, 2)

    # Plot setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(dpi=300, figsize=(10, 6))
    ax = sns.barplot(x=bin_centers, y=bin_counts, palette="viridis")

    # Customize plot
    ax.set_xlabel("Change in rewrite similarity", fontsize=14)
    ax.set_ylabel("Document count", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for i, count in enumerate(bin_counts):
        ax.text(i, count + 1, str(count), ha="center", va="bottom", fontsize=10)

    sns.despine()

    # Save figure
    fig_path = Path("..") / "figures" / "rewrite_diffs_in_zero_vocab_setting.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)


def rewrite_quality_og_vs_synonyms():
    """Calculate the difference between original descriptors and rewrites in the zero vocab setting."""
    path_to_og = (
        Path("..")
        / "results"
        / "final_zero_vocab"
        / "descriptors_final_zero_vocab.jsonl"
    )
    path_to_final = (
        Path("..")
        / "results"
        / "final_zero_vocab"
        / "descriptors_final_zero_vocab_final.jsonl"
    )

    og_scores = []
    final_scores = []

    with path_to_og.open("r") as f:
        for line in f:
            line = json.loads(line, strict=False)
            scores = line["similarity"]
            og_scores.append(max(scores))

    with path_to_final.open("r") as f:
        for line in f:
            line = json.loads(line, strict=False)
            final_scores.append(line["similarity"][0])

    # Write some basic stats to file
    basic_stats(og_scores, final_scores)

    # Plot score distribution
    plot_score_distribution(og_scores, final_scores)

    # Plot score differences
    plot_score_differences(og_scores, final_scores)


def plot_effect_of_revision_rounds():

    rewrite_scores = read_similarity_scores("og", return_all=True)

    num_plots = len(rewrite_scores)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, (run, scores) in enumerate(rewrite_scores.items()):
        max_indices = [np.argmax(sim) for sim in scores]
        count_dict = Counter(max_indices)
        x = list(count_dict.keys())
        y = list(count_dict.values())

        sns.barplot(x=x, y=y, palette="viridis", edgecolor="black", ax=axes[idx])
        axes[idx].set_title(run)
        axes[idx].set_xlabel("Best rewrite index")

        for i, val in zip(x, y):
            axes[idx].text(i, val, str(val), ha="center", va="bottom")

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("../figures/revision_effects_grid.png")


if __name__ == "__main__":
    calculate_avg_rewrite_scores()
    rewrite_quality_og_vs_synonyms()
    plot_effect_of_revision_rounds()

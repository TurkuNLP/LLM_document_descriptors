import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

runs = [
    "final_zero_vocab",
    "final_50_vocab",
    "final_100_vocab",
    "final_200_vocab",
    "final_300_vocab",
    "final_500_vocab",
]


def read_file(file_path):
    with open(file_path, "r") as file:
        return [float(line.strip()) for line in file]


# Group files based on naming scheme
grouped_files = {
    "no-vocab": [],
    "max-vocab-50": [],
    "max-vocab-100": [],
    "max-vocab-200": [],
    "max-vocab-300": [],
    "max-vocab-500": [],
}

for group, run in zip(grouped_files, runs):
    path = Path("..") / "results" / run / f"descriptor_count_growth_{run}.txt"
    grouped_files[group] = read_file(path)

# Plot the results
descriptive_labels = [
    "No vocabulary",
    "Max vocabulary 50",
    "Max vocabulary 100",
    "Max vocabulary 200",
    "Max vocabulary 300",
    "Max vocabulary 500",
]

colors = ["b", "orange", "g", "r", "purple"]
for idx, label in enumerate(grouped_files):
    # Plot average with solid line
    plt.plot(grouped_files[label], label=descriptive_labels[idx], color=colors[idx])

    # plt.xticks(ticks=range(20), labels=range(1, 50))
    plt.yticks(ticks=range(500, 4000, 500))

plt.xlabel("Batches (200 documents each)")
plt.ylabel("Unique (general) descriptors")
plt.title("Descriptor number growth")
plt.legend()
plt.grid(
    True, which="both", linestyle="--", linewidth=0.5, alpha=0.7
)  # Add faint gridlines
plt.save_fig("../figures/descriptor_count_growth")

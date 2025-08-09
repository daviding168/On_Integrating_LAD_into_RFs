import matplotlib.pyplot as plt
import numpy as np

# Different depths
datasets = ["max-depth=10", "max-depth=15", "max-depth=20"]

# Accuracy used for depth 10, 15, and 20
acc = {
    "RF_sklearn": [86.02, 86.11, 85.93],
    "RF_LAD_majority_vote": [82.35, 83.07, 82.48],
    "RF_LAD_soft_vote": [82.27, 82.84, 82.49]
}

# Features used for depth 10, 15, and 20
features_used = {
    "RF_sklearn": [748.13, 748.93, 748.57],
    "RF_LAD_majority_vote": [260.63, 263.7, 259.63],
    "RF_LAD_soft_vote": [259.7, 264.37, 263.06]
}

# Size AXp for depth 10, 15, and 20
size_axp = {
    "RF_sklearn": [243.39, 245.32, 247.34],
    "RF_LAD_majority_vote": [63.1, 67.54, 62.73],
    "RF_LAD_soft_vote": [68, 66, 68.55]
}

# Size CXp for depth 10, 15, and 20
size_cxp = {
    "RF_sklearn": [33.65, 36.74, 36.98],
    "RF_LAD_majority_vote": [10.18, 9.18, 7.38],
    "RF_LAD_soft_vote": [10.60, 8.98, 8.47]
}

# Time AXp for depth 10, 15, and 20
time_axp = {
    "RF_sklearn": [202.8971466666667, 54.517586666666645, 203.74548000000001],
    "RF_LAD_majority_vote": [0.65, 1.09, 1.40],
    "RF_LAD_soft_vote": [1.21, 1.17, 1.15]
}

# Time CXp for depth 10, 15, and 20
time_cxp = {
    "RF_sklearn": [0.8409, 0.8596, 0.8563],
    "RF_LAD_majority_vote": [0.1746, 0.2011, 0.1952],
    "RF_LAD_soft_vote": [0.1902, 0.2068, 0.2022]
}

# Set up subplots for each statistic
fig, axes = plt.subplots(6, 1, figsize=(25, 20), sharex=True)

# Bar style for RF_sklearn, RF_LAD_majority_vote, RF_LAD_soft_vote
patterns = ['///', '...', 'xxx']  # Different hatching styles for the bars

# Bar width and spacing
spacing_factor = 0.5
x = np.arange(len(datasets)) * spacing_factor
width = 0.05

# List of metrics and corresponding labels
metrics = [acc, features_used, size_axp, size_cxp, time_axp, time_cxp]
y_labels = [
    'Avg. accuracy (%)',
    'Avg. nb. of features',
    'Avg. size of AXps',
    'Avg. size of CXps',
    'Avg. time of AXps (s)',
    'Avg. time of CXps (s)'
]

# Plot each metric
for i, (metric, ylabel) in enumerate(zip(metrics, y_labels)):
    ax = axes[i]
    # Apply log scale **only** to the "Avg. time of AXps (s)" plot
    if ylabel == "Avg. time of AXps (s)":
        ax.set_yscale("log")
    ax.bar(x - width, metric["RF_sklearn"], width, label='RF-sklearn', hatch=patterns[0], color='white', edgecolor='black')
    ax.bar(x, metric["RF_LAD_majority_vote"], width, label='RF-LAD (majority-votes)', hatch=patterns[1], color='white', edgecolor='black')
    ax.bar(x + width, metric["RF_LAD_soft_vote"], width, label='RF-LAD (soft-votes)', hatch=patterns[2], color='white', edgecolor='black')
    ax.set_ylabel(ylabel, fontsize=14)

# Move legend to avoid overlap
axes[0].legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))

# Set x-axis labels and ticks
plt.xticks(x, datasets, rotation=0, fontsize=20)
# plt.xlabel("Max Depth", fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

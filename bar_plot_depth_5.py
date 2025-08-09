import matplotlib.pyplot as plt
import numpy as np

# Data for all datasets
datasets = [
    "ad", "anneal", "australian-credit", "breast-cancer", "cnae", "contraceptive", "drilling",
    "fetal-health", "german-credit", "heart-cleveland", "hepatitis", "hypothyroid", "indian-diabetes",
    "indian-liver", "loan", "lymph", "pd-speech", "pima", "reuters", "soybean", "spambase", "startup", "wdbc"
]

# Features used for all datasets
features_used = {
    "RF_sklearn": [354.44, 84.23, 122.47, 83.5, 209.66, 90, 56, 92.8, 109.9, 94.73, 68,
                   79.17, 96.9, 84, 67.8, 66.1, 682.16, 124.94, 244.37, 46.73, 217.4, 107.8, 87.7],
    "RF_LAD_majority_vote": [91.94, 50.24, 94.7, 77.47, 51.53, 71.6, 49.14, 84.8, 100.03, 85.4, 66,
                             49.24, 88.9, 80.5, 64.44, 61.93, 223.22, 63.6, 110.73, 40.24, 116.9, 100.33, 76.4],
    "RF_LAD_soft_vote": [91.4, 50.5, 95.27, 77.7, 50.97, 71.56, 49.93, 85.53, 100, 85, 66.1,
                         49.2, 88.6, 80.8, 64.36, 61.76, 223.47, 62.93, 111.17, 40.14, 120.47, 100.56, 77.06]
}

# Size AXp for all datasets
size_axp = {
    "RF_sklearn": [113.58, 22.12, 37.87, 27.64, 43.4, 39.42, 26.8, 32.75, 46.49, 32.16, 21.39,
                   11.53, 36.43, 29.2, 34.12, 24.69, 138.32, 45.06, 39.26, 8.41, 55.45, 41.21, 23.92],
    "RF_LAD_majority_vote": [14.77, 8.29, 16.2, 19.69, 10.27, 19.84, 19.03, 22.35, 34.16, 16.83, 17.75,
                             4.99, 22.18, 22.2, 14.98, 18.2, 36.68, 18, 4.54, 8.37, 12.69, 28.81, 21],
    "RF_LAD_soft_vote": [14.89, 10.33, 18.47, 16.24, 8.81, 18.18, 20.82, 20.26, 32.39, 18.47, 16.46,
                         4.15, 22.13, 24.1, 14.85, 17.77, 27.54, 16.38, 4.93, 7.81, 16.32, 28.45, 24.98]
}

# Size CXp for all datasets
size_cxp = {
    "RF_sklearn": [13.62, 10.06, 5.99, 7.75, 4.6, 7.33, 3.35, 11.75, 18.28, 7.4, 7.57,
                   8.03, 8.05, 13.38, 11.17, 4.72, 31.3, 19.04, 25.79, 6.24, 15.44, 10.08, 7.26],
    "RF_LAD_majority_vote": [1.42, 5.3, 4.96, 3.5, 1.23, 3.92, 2.26, 8.794, 11.05, 2.2, 5.74,
                             1.54, 5.1, 6.76, 9.04, 3.49, 10.77, 6.62, 1.12, 4.14, 4.08, 3.18, 6.6],
    "RF_LAD_soft_vote": [1.76, 1.21, 3.4, 2.63, 1.2, 4.51, 2.52, 4.206, 9.61, 2.03, 5.87,
                         1.8, 5.17, 6.84, 9.65, 3.38, 6.76, 6.55, 1.39, 4.19, 4.56, 4.53, 8.03]
}

# Set up subplots for each statistic
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

# Bar style for RF_sklearn, RF_LAD_majority_vote, RF_LAD_soft_vote
patterns = ['///', '...', 'xxx']  # Different hatching styles for the bars

# Plot Features Used
ax = axes[0]
spacing_factor = 5
x = np.arange(len(datasets)) * spacing_factor
width = 1.2
ax.bar(x - width, features_used["RF_sklearn"], width, label='RF-sklearn', hatch=patterns[0], color='white',
       edgecolor='black')
ax.bar(x, features_used["RF_LAD_majority_vote"], width, label='RF-LAD (majority-votes)', hatch=patterns[1],
       color='white',
       edgecolor='black')
ax.bar(x + width, features_used["RF_LAD_soft_vote"], width, label='RF-LAD (soft-votes)', hatch=patterns[2],
       color='white',
       edgecolor='black')
ax.set_ylabel('Avg. number of features', fontsize=20)
# ax.set_title('Features Used per Dataset')
ax.legend(fontsize=20)

# Plot Size AXp
ax = axes[1]
ax.bar(x - width, size_axp["RF_sklearn"], width, label='RF-sklearn', hatch=patterns[0], color='white',
       edgecolor='black')
ax.bar(x, size_axp["RF_LAD_majority_vote"], width, label='RF-LAD (majority-votes)', hatch=patterns[1], color='white',
       edgecolor='black')
ax.bar(x + width, size_axp["RF_LAD_soft_vote"], width, label='RF-LAD (soft_votes)', hatch=patterns[2], color='white',
       edgecolor='black')
ax.set_ylabel('Avg. size of AXps', fontsize=20)
# ax.set_title('Size AXp per Dataset')

# Plot Size CXp
ax = axes[2]
ax.bar(x - width, size_cxp["RF_sklearn"], width, label='RF-sklearn', hatch=patterns[0], color='white',
       edgecolor='black')
ax.bar(x, size_cxp["RF_LAD_majority_vote"], width, label='RF-LAD (majority-votes)', hatch=patterns[1], color='white',
       edgecolor='black')
ax.bar(x + width, size_cxp["RF_LAD_soft_vote"], width, label='RF-LAD (soft_votes)', hatch=patterns[2], color='white',
       edgecolor='black')
ax.set_ylabel('Avg. size of CXps', fontsize=20)
# ax.set_title('Size CXp per Dataset')

# Set x-axis labels and ticks
plt.xticks(x, datasets, rotation=45, fontsize=15.5)
plt.tight_layout()

# Show the plot
plt.show()

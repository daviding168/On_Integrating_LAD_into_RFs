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
    "RF_sklearn": [279.75, 81.67, 120.77, 80.2, 167.87, 89.7, 55.8, 92.14, 109.3, 94.36, 67.94, 77.47, 96.23, 84, 67.73,
                   65.86, 543.2, 119.44, 225.3, 45.93, 185.7, 106.86, 87.03],
    "RF_LAD_majority_vote": [68.36, 42.26, 88.9, 71.1, 36.93, 60.07, 47.87, 76.47, 97.17, 82, 64.73, 41.9, 85.67, 79.36,
                             62.3, 60.03, 190.2, 61.53, 88.44, 39.24, 88.44, 95.8, 74.86],
    "RF_LAD_soft_vote": [65.73, 42.2, 88.94, 71.33, 35.53, 59.36, 48.07, 76.36, 96.3, 81.56, 65.27, 41.8, 85.3, 79.14,
                         62.3, 60.03, 191.97, 61.53, 87.97, 39.13, 88.3, 95.33, 73.97]
}

# Size AXp for all datasets
size_axp = {
    "RF_sklearn": [91.2, 20.71, 28.79, 26.13, 34.88, 34.35, 24.77, 28.30, 34.04, 23.94, 19.62, 9.32, 28.26, 25.22,
                   34.55, 21.57, 85.9, 37.53, 26.97, 4.7, 42.71, 29.51, 20],
    "RF_LAD_majority_vote": [13.11, 6.21, 14.14, 14.67, 6.31, 12.15, 17.7, 12.41, 27.05, 16.42, 16.03, 4.16, 15.21,
                             18.92, 14.36, 15.38, 18.49, 12.95, 2.62, 6.31, 7.93, 24.28, 20.87],
    "RF_LAD_soft_vote": [11.07, 5.83, 15.26, 16.95, 7.59, 17.63, 19.7, 12.67, 23.13, 12.55, 15.9, 4.35, 14.94, 18.28,
                         9.92, 15.52, 20.05, 12.29, 3.42, 6.64, 7.94, 22.84, 22.92]
}

# Size CXp for all datasets
size_cxp = {
    "RF_sklearn": [12.72, 11.33, 6.3, 6.69, 4.55, 7.15, 3.84, 10.029, 22.94, 5.91, 8.37, 5.63, 8.19, 14.67, 9.27, 5.35,
                   30.32, 18.79, 33.45, 9.22, 15.14, 9.29, 6.48],
    "RF_LAD_majority_vote": [2.43, 2.39, 2.39, 1.88, 1.71, 3.17, 3.17, 4.32, 10.37, 3.05, 8.79, 1.68, 2.78, 7.28, 11.32,
                             3.42, 7.53, 2.48, 1.16, 4.11, 5.62, 2.69, 4],
    "RF_LAD_soft_vote": [1.48, 2.1, 5.83, 2.62, 1.15, 5.19, 2.91, 6.23, 11.34, 1.88, 8.62, 1.62, 4.07, 9.89, 8.97, 2.98,
                         6.94, 3.48, 1.96, 3.35, 4.62, 3.96, 6.01]
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
ax.bar(x, features_used["RF_LAD_majority_vote"], width, label='RF-LAD (majority-votes)', hatch=patterns[1], color='white',
       edgecolor='black')
ax.bar(x + width, features_used["RF_LAD_soft_vote"], width, label='RF-LAD (soft-votes)', hatch=patterns[2], color='white',
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

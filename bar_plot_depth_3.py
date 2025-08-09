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
    "RF_sklearn": [207.16, 76.7, 110.67, 73.14, 127, 85.67, 54.97, 85.47, 104.36, 90.06, 67.56, 69.6, 91.56, 83.03,
                   67.7, 63.86, 328.37, 107.23, 172.56, 44.13, 131.9, 97.47, 82.23],
    "RF_LAD_majority_vote": [46.8, 29.18, 75.47, 54.6, 22.76, 41.13, 43.07, 48.13, 86.27, 66.9,
                             62.24, 19.7, 70.86, 72.06, 56.43, 55.5, 138.7, 52.74, 48.14, 35.1, 48.43, 78.36, 66.17],
    "RF_LAD_soft_vote": [47.36, 29.82, 75.06, 53.37, 23.6, 40.86, 43.03, 48.24, 87.44, 66.53, 61.37, 19.63, 70.6, 73.77,
                         55.6, 55.76, 139.84, 53.07, 49.26, 35.36, 48.9, 78.23, 66.36]
}

# Size AXp for all datasets
size_axp = {
    "RF_sklearn": [77.13, 16.3, 21, 19.65, 25.8, 24.26, 22.45, 20.08, 22.14, 20.91,
                   14.89, 6.31, 18.21, 13.76, 22.65, 19.27, 44.39, 26.91, 7.36, 2.24, 33.97,
                   21.67, 17.58],
    "RF_LAD_majority_vote": [7.38, 2.47, 5.1, 11.79, 4.43, 9.61, 16.07, 6.184, 16.75, 10.28, 13.09,
                             1.23, 8.81, 14.14, 7.88, 10.37, 8.1, 5.52, 2.21, 3.81, 7.45, 13.48, 18.3],
    "RF_LAD_soft_vote": [6.37, 4.06, 9.46, 10.03, 4.92, 9.95, 14.57, 7.09, 18.41, 10.23, 12.79,
                         2.23, 7.69, 14.42, 8.45, 11.37, 10.22, 6.95, 2.21, 4.73, 7.46, 15.84, 13.03]
}

# Size CXp for all datasets
size_cxp = {
    "RF_sklearn": [17.56, 9.22, 6.29, 6.44, 5.89, 5.13, 3.01, 11.77, 20.16, 5.37, 9.2,
                   9.26, 7.94, 19.04, 15.82, 5.37, 29.24, 15.79, 27.15, 7.64, 14.95, 9.32, 6.4],
    "RF_LAD_majority_vote": [1.45, 2.19, 1.7, 2.58, 1.07, 2.53, 2.4, 2.27, 11.71, 3.05, 7.64,
                             1.36, 3.32, 10.3, 13.64, 4.4, 8.89, 2.64, 1.11, 4.39, 3.82, 5.76, 7.83],
    "RF_LAD_soft_vote": [1.28, 1, 1.93, 2.56, 1.28, 2.15, 2.47, 3.57, 17.7, 3.72, 6.42,
                         1.59, 2.66, 10.34, 12.69, 2.47, 8.47, 3.06, 1.11, 3.42, 3.25, 3.76, 3.89]
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

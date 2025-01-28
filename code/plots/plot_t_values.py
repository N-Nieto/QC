# %%
import seaborn as sbn
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa

results = pd.DataFrame()
sampling_list = ["low_Q", "high_Q"]
age_bins = 10
random_q_repetitions = 20
results_dir = "/output/statistics/"

# Initialize a dictionary to store T-values for each sampling strategy
t_values_dict = {}

# Read and store T-values for each sampling strategy
for sampling in sampling_list:
    results_sampling = pd.read_csv(project_root + results_dir + "statistic_test_" + str(age_bins) + "_bins_sampling_" + sampling + "_5_sites.csv", index_col=0)  # noqa
    t_values_dict[sampling] = np.abs(results_sampling["t-stat"])

# # Read and store T-values for the random distribution
# sampling = "random_Q"
# results_sampling = pd.read_csv(project_root + results_dir + "statistic_test_" + str(age_bins) + "bins_" + str(random_q_repetitions) + "repeated_random_sampling_5_sites.csv", index_col=0)  # noqa
# t_values_dict[sampling] = results_sampling["t-stats"]

# Plotting the T-value distributions
plt.figure(figsize=(10, 6))
for sampling, t_values in t_values_dict.items():
    plt.hist(t_values, bins=100, alpha=0.5, label=sampling)

plt.xlabel('T-value')
plt.ylabel('Frequency')
plt.title('Distribution of T-values for each sampling strategy')
plt.legend()
plt.show()
# %%

# %%
import pandas as pd

results = pd.DataFrame()
sampling_list = ["low_Q", "high_Q", "random_Q"]

results_dir ="/home/nnieto/Nico/Harmonization/QC/output/statistics/"
p_value_threshold=0.05
for sampling in sampling_list:

    results_sampling = pd.read_csv(results_dir+"sampling_"+sampling+".csv", index_col=0)
    results = pd.concat([results, results_sampling])

    significant_features_count = results_sampling[results_sampling["P-value"] < p_value_threshold].__len__()
    print("For "+sampling+": Number of statistically significant features: " + str(significant_features_count))
# %%

import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 15))
plt.yscale("log")

results["Significant"] = results["P-value"] < p_value_threshold
results = results[results["P-value"] < p_value_threshold]
sbn.boxenplot(data=results, y="P-value", x="sampling", hue="Significant")
plt.axhline(y=p_value_threshold, color='r', linestyle='--', label=f'Significance Threshold (p={p_value_threshold})')

# %%

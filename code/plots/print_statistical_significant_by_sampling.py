# %%
import pandas as pd
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa

results = pd.DataFrame()
sampling_list = ["low_Q", "high_Q"]
age_bins = 3
random_q_repetitions = 20
results_dir = "/output/refactor/statistics/"

p_value_threshold = 0.05/3747
print("With Bonferroni correction:")
for sampling in sampling_list:

    results_sampling = pd.read_csv(project_root+results_dir+"statistic_test_"+str(age_bins)+"_bins_sampling_"+sampling+".csv", index_col=0)       # noqa

    significant_features = results_sampling[results_sampling["P-value"] < p_value_threshold]                        # noqa

    print("For "+sampling+": median pvalue: " + str(significant_features["P-value"].median()))                      # noqa
    print("For "+sampling+": Number of statistically significant features: " + str(significant_features.__len__())) # noqa


# for the random distributions
sampling = "random_Q"
results_sampling = pd.read_csv(project_root+results_dir+"statistic_test_"+str(age_bins)+"bins_"+str(random_q_repetitions)+"repeated_random_sampling.csv", index_col=0)       # noqa

significant_features = results_sampling[results_sampling["P-value"] < p_value_threshold]                        # noqa

print("For "+sampling+": median pvalue: " + str(significant_features["P-value"].median()))                      # noqa
print("For "+sampling+": Number of statistically significant features: " + str(significant_features.__len__()/random_q_repetitions)) # noqa


# %%

# %%
import pandas as pd
import os
import seaborn as sbn
import matplotlib.pyplot as plt

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa

results_dir = "/output/refactor/ML/"

metric_to_plot = "AUC"

results_random = pd.read_csv(project_root+results_dir+"results_10_bins_random_20_repetitions_unconf_pooled_data_site_SALD_eNKI_CamCAN_all_sampling_Q.csv")                      # noqa
results_random.drop(columns=["Repeated"])
results_random["QC_Sampling"] = "random_Q"
# Load data pooled
results = pd.read_csv(project_root+results_dir+"results_10_bins_unconf_pooled_data_site_SALD_eNKI_CamCAN_all_sampling_Q.csv")                                                   # noqa
metric_to_plot = "AUC"
results = results[results["QC_Sampling"] != "random_Q"]
results = pd.concat([results, results_random])

results_pooled = results[results["Model"] == "Pooled data Test"]
results_pooled["Site"] = "Pooled"

# Load data
results = pd.read_csv(project_root+results_dir+"results_unconfound_10bins_single_site_SALD_eNKI_CamCAN_high_low_sampling_Q.csv")                                                # noqa
results = results[results["Model"] == "None Test"]

results_random = pd.read_csv(project_root+results_dir+"results_single_site_10_bins_random_20_repetitions_unconf_data_site_SALD_eNKI_CamCAN_high_low_sampling_Q.csv")            # noqa
results_random.drop(columns=["Repeated"])
results_random = results_random[results_random["Model"] == "Pooled data Test"]
results_random["QC_Sampling"] = "random_Q"
results_random.drop(columns=["Model"])
results.drop(columns=["Model"])

results_test = pd.concat([results, results_random, results_pooled])
results_test.QC_Sampling.replace({"low_Q": "low Q",
                                  "random_Q": "random Q",
                                  "high_Q": "high Q"}, inplace=True)
results_test.rename(columns={"QC_Sampling": "Q Sampling"}, inplace=True)
order = ["low Q", "random Q", "high Q"]
plt.figure(figsize=(17, 17))
plt.subplot(2, 1, 1)

sbn.boxplot(data=results_test, x="Site",
            y=metric_to_plot, hue="Q Sampling",
            hue_order=order, palette=sbn.color_palette("tab10")
            )
plt.grid()
plt.ylim([0.35, 1])
plt.xlabel("Training data")
plt.title("A", loc="left")

plt.ylabel("Test AUC")

# 3 age bins
metric_to_plot = "AUC"

results_random = pd.read_csv(project_root+results_dir+"results_3_bins_random_20_repetitions_N_balanced_unconf_pooled_data_site_SALD_eNKI_CamCAN_all_sampling_Q.csv")             # noqa
results_random.drop(columns=["Repeated"])
results_random["QC_Sampling"] = "random_Q"

# Load data pooled
results = pd.read_csv(project_root+results_dir+"results_3_bins_unconf_pooled_data_site_SALD_eNKI_CamCAN_all_sampling_Q.csv")             # noqa
metric_to_plot = "AUC"
results = results[results["QC_Sampling"] != "random_Q"]

results = pd.concat([results, results_random])

results_pooled = results[results["Model"] == "Pooled data Test"]

results_pooled["Site"] = "Pooled"

# Load data
results = pd.read_csv(project_root+results_dir+"results_unconfound_3bins_single_site_SALD_eNKI_CamCAN_high_low_sampling_Q.csv")             # noqa
results = results[results["Model"] == "None Test"]

results_random = pd.read_csv(project_root+results_dir+"results_single_site_3_bins_random_20_repetitions_unconf_data_site_SALD_eNKI_CamCAN_high_low_sampling_Q.csv")             # noqa
results_random.drop(columns=["Repeated"])
results_random = results_random[results_random["Model"] == "Pooled data Test"]
results_random["QC_Sampling"] = "random_Q"
results_random.drop(columns=["Model"])
results.drop(columns=["Model"])

results_test = pd.concat([results, results_random, results_pooled])
results_test.QC_Sampling.replace({"low_Q": "low Q",
                                  "random_Q": "random Q",
                                  "high_Q": "high Q"}, inplace=True)
results_test.rename(columns={"QC_Sampling": "Q Sampling"}, inplace=True)

order = ["low Q", "random Q", "high Q"]
plt.subplot(2, 1, 2)
sbn.boxplot(data=results_test, x="Site",
            y=metric_to_plot, hue="Q Sampling", dodge=True,
            hue_order=order, palette=sbn.color_palette("tab10")
            )
plt.ylim([0.35, 1])
plt.title("B", loc="left")
plt.grid()
plt.xlabel("Training data")
plt.ylabel("Test AUC")
# %%

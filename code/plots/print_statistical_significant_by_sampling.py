# %%
import seaborn as sbn
import pandas as pd
from pathlib import Path

project_root = Path().resolve().parents[1]


results = pd.DataFrame()
sampling_list = ["low_Q", "high_Q"]
age_bins = 10
random_q_repetitions = 20
results_dir = project_root / "output" / "statistics/"

p_value_threshold = 0.05 / 3747
print("With Bonferroni correction:")
data_10 = []

stats_all = []

for sampling in sampling_list:
    results_sampling = pd.read_csv(
        results_dir
        / (
            "statistic_test_"
            + str(age_bins)
            + "_bins_sampling_"
            + sampling
            + "_5_sites.csv"
        ),
        index_col=0,
    )

    significant_features = results_sampling[
        results_sampling["P-value"] < p_value_threshold
    ]

    print(
        "For "
        + sampling
        + ": median pvalue: "
        + str(significant_features["P-value"].median())
    )
    print(
        "For "
        + sampling
        + ": Number of statistically significant features: "
        + str(significant_features.__len__())
    )

    data_10.append(
        {
            "Sampling_Q": sampling,
            "Median p-value": significant_features["P-value"].median(),  # noqa
            "Number of Significant features": significant_features.__len__(),
        }
    )  # noqa

# %%
# # for the random distributions
sampling = "random_Q"
results_sampling = pd.read_csv(
    results_dir
    / (
        "statistic_test_"
        + str(age_bins)
        + "bins_"
        + str(random_q_repetitions)
        + "repeated_random_sampling_5_sites.csv"
    ),
    index_col=0,
)  # noqa

significant_features = results_sampling[results_sampling["P-value"] < p_value_threshold]  # noqa

print(
    "For "
    + sampling
    + ": median pvalue: "
    + str(significant_features["P-value"].median())
)  # noqa
print(
    "For "
    + sampling
    + ": Number of statistically significant features: "
    + str(significant_features.__len__() / random_q_repetitions)
)  # noqa

data_10.append(
    {
        "Sampling_Q": sampling,
        "Median p-value": significant_features["P-value"].median(),
        "Number of Significant features": significant_features.__len__()
        / random_q_repetitions,
    }
)  # noqa
data_10 = pd.DataFrame(data_10)

data_10["Age bins"] = 10

# %%
age_bins = 3
random_q_repetitions = 20

p_value_threshold = 0.05 / 3747
print("With Bonferroni correction:")
data_3 = []
for sampling in sampling_list:
    results_sampling = pd.read_csv(
        results_dir
        / (
            "statistic_test_"
            + str(age_bins)
            + "_bins_sampling_"
            + sampling
            + "_5_sites.csv"
        ),
        index_col=0,
    )  # noqa

    significant_features = results_sampling[
        results_sampling["P-value"] < p_value_threshold
    ]  # noqa

    print(
        "For "
        + sampling
        + ": median pvalue: "
        + str(significant_features["P-value"].median())
    )  # noqa
    print(
        "For "
        + sampling
        + ": Number of statistically significant features: "
        + str(significant_features.__len__())
    )  # noqa
    data_3.append(
        {
            "Sampling_Q": sampling,
            "Median p-value": significant_features["P-value"].median(),  # noqa
            "Number of Significant features": significant_features.__len__(),
        }
    )  # noqa
# %%
# # for the random distributions
sampling = "random_Q"
results_sampling = pd.read_csv(
    results_dir
    / (
        "statistic_test_"
        + str(age_bins)
        + "bins_"
        + str(random_q_repetitions)
        + "repeated_random_sampling_5_sites.csv"
    ),
    index_col=0,
)  # noqa

significant_features = results_sampling[results_sampling["P-value"] < p_value_threshold]  # noqa

print(
    "For "
    + sampling
    + ": median pvalue: "
    + str(significant_features["P-value"].median())
)  # noqa
print(
    "For "
    + sampling
    + ": Number of statistically significant features: "
    + str(significant_features.__len__() / random_q_repetitions)
)  # noqa

data_3.append(
    {
        "Sampling_Q": sampling,
        "Median p-value": significant_features["P-value"].median(),  # noqa
        "Number of Significant features": significant_features.__len__()
        / random_q_repetitions,
    }
)  # noqa
data_3 = pd.DataFrame(data_3)

data_3["Age bins"] = 3
# %%

data = pd.concat([data_3, data_10])
# %%

# %%
import matplotlib.pyplot as plt
import seaborn as sbn

df = data
df["Sampling_Q"].replace(
    {
        "low_Q": "Low Quality",
        "high_Q": "High Quality",
        "random_Q": "Random Quality",
    },
    inplace=True,
)
# Define the desired order and hue_order
order = ["Low Quality", "Random Quality", "High Quality"]
hue_order = [10, 3]

# Initialize the plot
plt.figure(figsize=(10, 8))
ax = sbn.barplot(
    data=df,
    x="Sampling_Q",
    y="Number of Significant features",
    hue="Age bins",
    order=order,  # Enforce category order
    hue_order=hue_order,  # Enforce hue order
    palette=["yellowgreen", "red"],  # 10: green, 3: red
)

# Annotate p-values
for i, (cat, hue) in enumerate(zip(df["Sampling_Q"], df["Age bins"])):
    # Find the correct index in order/hue_order
    x_index = order.index(cat)  # x position
    hue_index = hue_order.index(hue)  # hue position

    # Calculate bar position
    bar_width = 0.8 / len(hue_order)  # Width split between hues
    x_pos = (
        x_index - 0.4 + (hue_index + 0.5) * bar_width
    )  # Adjust x-pos for hue   # noqa
    # Annotate above the correct bar
    height = df["Number of Significant features"].iloc[i]
    p_value = df["Median p-value"].iloc[i]
    ax.text(
        x_pos,
        height + 10,
        f"p={p_value:.1e}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )

# Customize the plot
plt.title(
    "Number of Significant Features [median p-value] by Sample Size and Quality",
    fontsize=12,
)  # noqa
plt.xlabel("Quality", fontsize=10)
plt.ylabel("Number of Significant features", fontsize=10)
plt.legend(title="Age bins")
plt.ylim([0, 700])

# Show the plot
plt.tight_layout()
plt.show()


# %%

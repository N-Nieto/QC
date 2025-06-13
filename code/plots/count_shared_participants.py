# %%
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_venn import venn2

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))

from lib.data_processing import (  # noqa
    get_min_common_number_images_in_age_bins,
    filter_age_bins_with_qc,
)  # noqa
from lib.data_loading import load_data_and_qc  # noqa
from lib.data_processing import (  # noqa
    keep_desired_age_range,
    get_age_bins,
    load_optimal_age_cuts,
)  # noqa


site_list = [
    "AOMIC_ID1000",
    "AOMIC-PIOP2",
    "AOMIC-PIOP1",
    "eNKI",
    "CamCAN",
    "SALD",
    "1000Brains",
    "GSP",
    "DLBS",
]

# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 3
age_cutoffs = load_optimal_age_cuts(project_root, site_list, n_age_bins)

print(f"Number of age bins: {n_age_bins}")
for row, site in enumerate(site_list):
    LOW_CUT_AGE = age_cutoffs[site]["low"]
    HICH_CUT_AGE = age_cutoffs[site]["high"]
    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)
    Y = keep_desired_age_range(Y, LOW_CUT_AGE, HICH_CUT_AGE)

    age_bins = get_age_bins(Y, n_age_bins)

    # Determine what is the max number of images in the
    # formed age bins for each gender
    n_images = get_min_common_number_images_in_age_bins(Y, age_bins)
    # n_images = 10
    # get the images depending the QC
    index_low = filter_age_bins_with_qc(Y, age_bins, n_images, sampling="low_Q")

    # get the images depending the QC
    index_high = filter_age_bins_with_qc(Y, age_bins, n_images, sampling="high_Q")

    median_qc_low = Y.loc[index_low, ["IQR"]].median().values[0]
    median_qc_high = Y.loc[index_high, ["IQR"]].median().values[0]
    # Convert arrays to sets
    set1 = set(index_low)
    set2 = set(index_high)

    # Print the number interception participants
    print(
        f"Site: {site}, N={index_low.__len__()}, Share: {len(set1.intersection(set2))}, ({100 * (len(set1.intersection(set2)) / len(set1)):.2f}%), \t  Difference of IQR median: {median_qc_low - median_qc_high:.3f}"
    )

    # # Create Venn diagram
    # plt.figure(figsize=(8, 6))
    # venn2([set1, set2], ("Low_QC", "High_QC"))
    # plt.title(
    #     "Number of Shared participants across QC sampling strategies for " + str(site)
    # )  # noqa
    # plt.show()
# %%
site_list = [
    "AOMIC_ID1000",
    "AOMIC-PIOP2",
    "AOMIC-PIOP1",
    "eNKI",
    "CamCAN",
    "SALD",
    "1000Brains",
    "GSP",
    "DLBS",
]
import pandas as pd
n_age_bins = 3

LOW_CUT_AGE = 18
HICH_CUT_AGE = 80
# Load data and prepare it
X_pooled = pd.DataFrame()
Y_pooled = pd.DataFrame()
for site in site_list:
    print(site)
    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)

    X_pooled = pd.concat([X_pooled, X], ignore_index=True)
    Y_pooled = pd.concat([Y_pooled, Y], ignore_index=True)
    Y_pooled["site"] = site

Y_pooled = keep_desired_age_range(Y_pooled, LOW_CUT_AGE, HICH_CUT_AGE)

age_bins = get_age_bins(Y_pooled, n_age_bins)

# Determine what is the max number of images in the
# formed age bins for each gender
n_images = get_min_common_number_images_in_age_bins(Y_pooled, age_bins)
# n_images = 10
# get the images depending the QC
index_low = filter_age_bins_with_qc(Y_pooled, age_bins, n_images, sampling="low_Q")

# get the images depending the QC
index_high = filter_age_bins_with_qc(Y_pooled, age_bins, n_images, sampling="high_Q")

median_qc_low = Y_pooled.loc[index_low, ["IQR"]].median().values[0]
median_qc_high = Y_pooled.loc[index_high, ["IQR"]].median().values[0]
# Convert arrays to sets
set1 = set(index_low)
set2 = set(index_high)
site = "Pooled_Data"
# Print the number interception participants
print(
    f"Site: {site}, N={index_low.__len__()}, Share: {len(set1.intersection(set2))}, ({100 * (len(set1.intersection(set2)) / len(set1)):.2f}%), \t  Difference of IQR median: {median_qc_low - median_qc_high:.3f}"
)

# %%

import seaborn as sns
plt.figure(figsize=(8, 6))

sns.histplot(
    Y_pooled,
    x="IQR",
    hue="gender",
    hue_order=["M", "F"],
    kde=True,
    element="step",
    stat="count",
    common_norm=False,
    bins=50,
)

# %%

sns.histplot(
    Y_pooled.loc[index_low, :],
    x="IQR",
    hue="gender",
    hue_order=["M", "F"],
    kde=True,
    element="step",
    stat="count",
    common_norm=False,
    bins=50,
)
# %%
sns.histplot(
    Y_pooled.loc[index_high, :],
    x="IQR",
    hue="gender",
    hue_order=["M", "F"],
    kde=True,
    element="step",
    stat="count",
    common_norm=False,
    bins=50,
)
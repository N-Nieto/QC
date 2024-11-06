# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
dir_path = '../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.data_processing import get_min_common_number_images_in_age_bins, filter_age_bins_with_qc # noqa
from lib.data_loading import load_ROI_data_and_qc                           # noqa
from lib.data_processing import keep_desired_age_range, get_age_bins          # noqa
from lib.ml import results_to_df_qc_single_site, results_qc_single_site                  # noqa
from scipy.stats import ttest_ind
# Sample Data (replace with your actual dataset)
# X = pd.DataFrame(...)  # Your feature matrix
# y = pd.Series(...)     # Your continuous target variable (age)

# Threshold for statistical significance (p-value)
p_value_threshold = 0.05

# ----------------------------
# 1. Initialize lists to store results
# ----------------------------
p_values = []  # To store p-values for each feature
significant_features_count = 0  # Counter for significant features


# Directions
data_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/region/"
save_dir = "/home/nnieto/Nico/Harmonization/QC/output/statistics/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]
# site_list = ["SALD"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10

clf = LogisticRegression()

results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
# random dosen't care about QC
sampling_list = ["low_Q", "high_Q", "random_Q"]

# plt.figure(figsize=(15, 8))
def confound_regression(X, Y, TIV):
    """
    Perform confound regression to remove the effect of confound variable TIV from the features X and target Y.

    Parameters:
    X (pd.DataFrame or np.ndarray): Input features with shape (n_samples, n_features)
    Y (pd.Series or np.ndarray): Target variable with shape (n_samples,)
    TIV (pd.Series or np.ndarray): Confound variable with shape (n_samples,)

    Returns:
    X_residual (np.ndarray): Features after removing the effect of TIV.
    Y_residual (np.ndarray): Target after removing the effect of TIV.
    """
    # Ensure TIV is in the correct shape (n_samples, 1)
    TIV = TIV.reshape(-1, 1) if len(TIV.shape) == 1 else TIV

    # Initialize the linear regression model
    reg = LinearRegression()

    # Regress each feature in X on TIV and compute the residuals
    X_residual = np.zeros_like(X)
    for i in range(X.shape[1]):
        reg.fit(TIV, X[:, i])  # Regress feature i on TIV
        X_residual[:, i] = X[:, i] - reg.predict(TIV)  # Subtract predicted values from original feature

    # Regress Y on TIV and compute the residuals
    reg.fit(TIV, Y)
    Y_residual = Y - reg.predict(TIV)

    return X_residual, Y_residual


for col, sampling in enumerate(sampling_list):
    print(sampling)
    X_pooled = pd.DataFrame()
    Y_pooled = pd.DataFrame()
    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_ROI_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)

        Y = keep_desired_age_range(Y, low_cut_age, high_cut_age)

        age_bins = get_age_bins(Y, n_age_bins)

        # Determine what is the max number of images in the
        # formed age bins for each gender
        n_images = get_min_common_number_images_in_age_bins(Y, age_bins)

        # get the images depending the QC
        index = filter_age_bins_with_qc(Y, age_bins,
                                        n_images, sampling=sampling)

        # filter the data
        Y = Y.loc[index]
        X = X.loc[index]
        X_pooled = pd.concat([X_pooled, X])
        Y_pooled = pd.concat([Y_pooled, Y])

    Y_pooled["gender"].replace({"F": 0, "M": 1}, inplace=True)
    X_pooled.drop(columns={"index"}, inplace=True)

    sites = Y_pooled["site"].reset_index()
    sites = sites["site"]
    if Y_pooled["TIV"].isna().sum() != 0:
        Y_pooled["TIV"].replace({np.nan: Y_pooled["TIV"].mean()}, inplace=True)

    Y = Y_pooled["gender"]
    X = X_pooled
    Y = Y.to_numpy()

    # Example usage:
    # Assuming X is your dataset with 3747 features, Y is your target, and TIV is your confound variable
    # X, Y, and TIV should be NumPy arrays or Pandas data structures
    X_residual, Y_residual = confound_regression(X.to_numpy(), Y, Y_pooled["TIV"].astype(float).to_numpy())

    X = pd.DataFrame(X_residual)
    feature_names = X.columns  # Get feature names

    p_values = []  # To store p-values for each feature
    significant_features_count = 0  # Counter for significant features
    for feature in feature_names:
        # Split the feature data into two groups based on the binary target (e.g., male vs. female)
        group1 = X[Y == 0][feature]  # Group corresponding to target == 0
        group2 = X[Y == 1][feature]  # Group corresponding to target == 1

        # Perform independent t-test
        t_stat, p_val = ttest_ind(group1, group2)

        # Append the p-value for this feature
        p_values.append(p_val)

        # Check if the feature is statistically significant
        if p_val < p_value_threshold:
            significant_features_count += 1

    # ----------------------------
    # 4. Convert p-values to a Pandas DataFrame for easy handling
    # ----------------------------
    p_values_df = pd.DataFrame({
        'Feature': feature_names,
        'P-value': p_values,
        'sampling': sampling
    })

    p_values_df.to_csv(save_dir+"ROI_unconfounded_10_bins_sampling_"+sampling+".csv")
# Add a red line to show the significance threshold (e.g., p=0.05)

print("Done!")

# %%

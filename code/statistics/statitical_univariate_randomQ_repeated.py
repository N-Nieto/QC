# %%
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)
from lib.data_processing import balance_data_age_gender_Qsampling       # noqa
from lib.data_loading import load_data_and_qc                           # noqa
from lib.data_processing import ConfoundRegressor_TIV                   # noqa

p_values = []  # To store p-values for each feature

# Directions (Indicate the sub folder inside /data)
# the folder data is assumed to be in the same herarchy as the project folder
data_dir = "/final_data_split/"
qc_dir = "/qc/"
save_dir = "/output/statistics/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10
random_q_repeated = 2

# randomly select participants
sampling = "random_Q"

confound_regressor = ConfoundRegressor_TIV()

for repeated in range(random_q_repeated):
    print(repeated)
    X_pooled = pd.DataFrame()
    Y_pooled = pd.DataFrame()
    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)
        # This is the main function to obtain different cohorts from the data
        X, Y = balance_data_age_gender_Qsampling(X, Y, n_age_bins, sampling)

        X_pooled = pd.concat([X_pooled, X])
        Y_pooled = pd.concat([Y_pooled, Y])

    # Replace the gender to number
    Y_pooled["gender"].replace({"F": 0, "M": 1}, inplace=True)
    # If there are any missing values replace is with the mean of the
    # TIV (There are not many missing)
    Y_pooled["TIV"].replace({np.nan: Y_pooled["TIV"].mean()}, inplace=True)
    TIV = Y_pooled["TIV"].astype(float).to_numpy()
    Y = Y_pooled["gender"].to_numpy()
    X = X_pooled

    # Remove TIV as confound
    confound_regressor.fit(X.to_numpy(), Y, TIV)
    X_residual, Y_residual = confound_regressor.transform(X.to_numpy(), Y, TIV)

    X = pd.DataFrame(X_residual)
    p_values = []  # To store p-values for each feature
    # Test for each feature if the gender distribution
    # of the features are different
    for feature in X.columns:
        # Split the feature data into two groups based on the binary target
        # (e.g., male vs. female)
        group1 = X[Y == 0][feature]  # Group corresponding to target == 0
        group2 = X[Y == 1][feature]  # Group corresponding to target == 1

        # Perform independent t-test
        t_stat, p_val = ttest_ind(group1, group2)

        # Append the p-value for this feature
        p_values.append(p_val)
    # ----------------------------
    # 4. Convert p-values to a Pandas DataFrame for easy handling and saving
    # ----------------------------
    if repeated == 0:
        p_values_df = pd.DataFrame({
            'Feature': X.columns,
            'P-value': p_values,
            'sampling': sampling
        })
    else:
        p_values_df_loop = pd.DataFrame({
            'Feature': X.columns,
            'P-value': p_values,
            'sampling': sampling
        })
        p_values_df = pd.concat([p_values_df, p_values_df_loop])

# %%
p_values_df.to_csv(project_root+save_dir+"statistic_test_"+str(n_age_bins)+"bins_"+str(random_q_repeated)+"repeated_random_sampling.csv")     # noqa
# Add a red line to show the significance threshold (e.g., p=0.05)

print("Experiment Done!")
# %%

# %%
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind
import os
from neuroHarmonize import harmonizationLearn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)
from lib.data_processing import balance_data_age_gender_Qsampling   # noqa
from lib.data_loading import load_data_and_qc                       # noqa
from lib.data_processing import ConfoundRegressor_TIV               # noqa

p_values = []  # To store p-values for each feature

save_dir = "/output/statistics/"

# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]
site_list = ["SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains"]


# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10

# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
sampling_list = ["low_Q", "high_Q"]

confound_regressor = ConfoundRegressor_TIV()

for col, sampling in enumerate(sampling_list):
    print(sampling)
    # Create a dataframe to pool the data from different sites
    # in each of the sampling schemes
    X_pooled = pd.DataFrame()
    Y_pooled = pd.DataFrame()
    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_data_and_qc(site=site)

        # This is the main function to obtain different cohorts from the data
        X, Y = balance_data_age_gender_Qsampling(X, Y, n_age_bins, sampling)

        X_pooled = pd.concat([X_pooled, X])
        Y_pooled = pd.concat([Y_pooled, Y])
        Y_pooled["site"] = row

    Y_pooled["gender"] = Y_pooled["gender"].replace({"F": 0, "M": 1}).astype(int)       # noqa
    # If there are any missing values replace is with the mean of the
    # TIV (There are not many missing)
    Y_pooled["TIV"].replace({np.nan: Y_pooled["TIV"].mean()}, inplace=True)
    covars = pd.DataFrame(Y_pooled["site"].to_numpy(), columns=['SITE'])

    TIV = Y_pooled["TIV"].astype(float).to_numpy()
    Y = Y_pooled["gender"].to_numpy()
    X = X_pooled.to_numpy()
    covars['Target'] = Y.ravel()
    harm_cheat, X = harmonizationLearn(data=X,
                                       covars=covars)
    # Remove TIV as confound
    confound_regressor.fit(X, Y, TIV)
    X_residual, Y_residual = confound_regressor.transform(X, Y, TIV)

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
    p_values_df = pd.DataFrame({
        'Feature': X.columns,
        'P-value': p_values,
        'sampling': sampling
    })

    # Save the results for each sampling
    p_values_df.to_csv(project_root+save_dir+"statistic_test_W_Harmonization_"+str(n_age_bins)+"_bins_sampling_"+sampling+"_5_sites.csv")    # noqa

print("Experiment Done!")

# %%

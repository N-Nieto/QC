# %%
import sys
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

pd.set_option('future.no_silent_downcasting', True)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)

from lib.data_loading import load_data_and_qc                                                       # noqa
from lib.data_processing import balance_data_age_gender_Qsampling, ConfoundRegressor_TIV                                               # noqa
from lib.ml import classification_results_by_site                                                   # noqa


# Directions (Indicate the sub folder inside /data)
# the folder data is assumed to be in the same herarchy as the project folder
data_dir = "/final_data_split/"
qc_dir = "/qc/"
save_dir = "/output/refactor/ML/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 3

clf = LogisticRegression()
confound_regressor = ConfoundRegressor_TIV()
kf_out = RepeatedStratifiedKFold(n_splits=2,
                                 n_repeats=2,
                                 random_state=23)

# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
sampling_list = ["low_Q", "high_Q"]
results = []

for col, sampling in enumerate(sampling_list):
    print(sampling)
    X_pooled = pd.DataFrame()
    Y_pooled = pd.DataFrame()
    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)

        # This is the main function to obtain different cohorts from the data
        X, Y = balance_data_age_gender_Qsampling(X, Y, n_age_bins, sampling)
        # Concatenate data from different sites
        X_pooled = pd.concat([X_pooled, X])
        Y_pooled = pd.concat([Y_pooled, Y])

    Y_pooled["gender"] = Y_pooled["gender"].replace({"F": 0, "M": 1}).astype(int)       # noqa
    Y_pooled["TIV"] = Y_pooled["TIV"].replace({np.nan: Y_pooled["TIV"].mean()})

    sites = Y_pooled["site"].reset_index()
    sites = sites["site"]
    data = X_pooled.copy()
    TIV = Y_pooled["TIV"].to_numpy()
    Y = Y_pooled["gender"]
    X = X_pooled.to_numpy()
    Y = Y.to_numpy()

    # Main loop
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
        print("FOLD: " + str(i_fold))

        # Patients used for train and internal XGB validation
        X_train = X[train_index, :]
        Y_train = Y[train_index]
        TIV_train = TIV[train_index]
        site_train = sites.iloc[train_index]

        # Patients used to generete a prediction
        X_test = X[test_index, :]
        Y_test = Y[test_index]
        site_test = sites.iloc[test_index]
        TIV_test = TIV[test_index]

        confound_regressor.fit(X_train, Y_train, TIV_train)
        X_residual, Y_residual = confound_regressor.transform(X_train, Y_train,
                                                              TIV_train)

        X_test, Y_residual = confound_regressor.transform(X_test, Y_test,
                                                          TIV_test)

        # None model
        clf.fit(X_residual, Y_train)
        pred_test = clf.predict_proba(X_test)[:, 1]
        results = classification_results_by_site(i_fold, "Pooled data Test", pred_test, Y_test, results, sampling, site_test)                 # noqa

        pred_train = clf.predict_proba(X_residual)[:, 1]
        results = classification_results_by_site(i_fold, "Pooled data Train", pred_train, Y_train, results, sampling, site_train)                 # noqa


# create a DataFrame for easier handeling
results = pd.DataFrame(results,
                       columns=["Fold",
                                "Model",
                                "Balanced ACC",
                                "AUC",
                                "F1",
                                "Recall",
                                "QC_Sampling",
                                "Site"
                                ])
# %%
results.to_csv(project_root+save_dir+"results_pooled_data_"+str(n_age_bins)+"_bins_high_low_sampling_Q.csv")   # noqa
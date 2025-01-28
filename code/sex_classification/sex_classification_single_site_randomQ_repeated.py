# %%
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)

from lib.data_processing import get_min_common_number_images_in_age_bins            # noqa
from lib.data_processing import filter_age_bins_with_qc                             # noqa
from lib.data_loading import load_data_and_qc                                       # noqa
from lib.data_processing import ConfoundRegressor_TIV                               # noqa
from lib.data_processing import keep_desired_age_range, get_age_bins                # noqa
from lib.ml import classification_results_by_site                         # noqa

save_dir = "/output/ML/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]
site_list = ["AOMIC_ID1000", "1000Brains"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10

clf = LogisticRegression()
confound_regressor = ConfoundRegressor_TIV()
kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)


# random dosen't care about QC
sampling = "random_Q"

random_q_repeated = 20

results = []

IQR_mean_loop = []
IQR_std_loop = []
IQR_meadian_loop = []
site_loop = []
for repeated in range(random_q_repeated):
    print("Repetition")
    print(repeated)

    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_data_and_qc(site=site)

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

        Y["gender"] = Y["gender"].replace({"F": 0, "M": 1}).astype(int)
        Y["TIV"] = Y["TIV"].replace({np.nan: Y["TIV"].mean()})

        TIV = Y["TIV"].to_numpy()
        Y = Y["gender"]
        X = X.to_numpy()
        Y = Y.to_numpy()

        # Main loop
        for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
            # print("FOLD: " + str(i_fold))

            # Patients used for train and internal XGB validation
            X_train = X[train_index, :]
            Y_train = Y[train_index]
            TIV_train = TIV[train_index]

            # Patients used to generete a prediction
            X_test = X[test_index, :]
            Y_test = Y[test_index]
            TIV_test = TIV[test_index]

            confound_regressor.fit(X_train, Y_train, TIV_train)
            X_residual, Y_residual = confound_regressor.transform(X_train,
                                                                  Y_train,
                                                                  TIV_train)

            X_test, Y_residual = confound_regressor.transform(X_test,
                                                              Y_test,
                                                              TIV_test)

            # None model
            clf.fit(X_residual, Y_train)
            pred_test = clf.predict_proba(X_test)[:, 1]
            results = classification_results_by_site(i_fold, "Single site Test", pred_test, Y_test, results, sampling, [site], repeated)                 # noqa

            pred_train = clf.predict_proba(X_residual)[:, 1]
            results = classification_results_by_site(i_fold, "Single site Train", pred_train, Y_train, results, sampling, [site], repeated)                 # noqa

# create a DataFrame for easier handeling
results = pd.DataFrame(results,
                       columns=["Fold",
                                "Model",
                                "Balanced ACC",
                                "AUC",
                                "F1",
                                "Recall",
                                "QC_Sampling",
                                "Site",
                                "Repeated"
                                ])
# %%
results.to_csv(project_root+save_dir+"results_single_site_"+str(n_age_bins)+"bins_random_"+str(random_q_repeated)+"repetitions_random_Q_AOMICID1000_10000brains.csv")   # noqa

# %%

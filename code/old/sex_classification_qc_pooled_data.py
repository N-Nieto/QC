# %%
import sys
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

dir_path = '../../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.data_processing import get_min_common_number_images_in_age_bins, filter_age_bins_with_qc # noqa
from lib.data_loading import load_data_and_qc                           # noqa
from lib.data_processing import keep_desired_age_range, get_age_bins          # noqa
from lib.ml import results_to_df_qc_single_site, results_qc_multiple_site                  # noqa

# Directions
data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
save_dir = "/home/nnieto/Nico/Harmonization/QC/output/sex_classification/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]
# site_list = ["SALD"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 3

clf = LogisticRegression()
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
# random dosen't care about QC
sampling_list = ["low_Q", "high_Q", "random_Q"]

X_pooled = pd.DataFrame()
Y_pooled = pd.DataFrame()
for col, sampling in enumerate(sampling_list):
    print(sampling)

    for row, site in enumerate(site_list):

        print(site)
        # Load data and prepare it
        X, Y = load_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)

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

    # Y_pooled["gender"].replace({"F": 0, "M": 1}, inplace=True)
    # sites = Y_pooled["site"].reset_index()
    # sites = sites["site"]
    # data = pd.concat([X_pooled, Y_pooled])
    print("QRC Mean Median STD")
    print(Y_pooled["IQR"].mean())
    print(Y_pooled["IQR"].median())
    print(Y_pooled["IQR"].std())
# %%
    Y = Y_pooled["gender"]
    X = X_pooled.to_numpy()
    Y = Y.to_numpy()

    # Main loop
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
        print("FOLD: " + str(i_fold))

        # Patients used for train and internal XGB validation
        X_train = X[train_index, :]
        Y_train = Y[train_index]
        site_train = sites.iloc[train_index]

        # Patients used to generete a prediction
        X_test = X[test_index, :]
        Y_test = Y[test_index]
        site_test = sites.iloc[test_index]
        # None model
        clf.fit(X_train, Y_train)
        pred_test = clf.predict_proba(X_test)[:, 1]
        results = results_qc_multiple_site(i_fold, "Pooled data Test", pred_test, Y_test, results, sampling, site_test)                 # noqa

        pred_train = clf.predict_proba(X_train)[:, 1]
        results = results_qc_multiple_site(i_fold, "Pooled data Train", pred_train, Y_train, results, sampling, site_train)                 # noqa


results = results_to_df_qc_single_site(results)
# %%
results.to_csv(save_dir+"results_pooled_data_site_SALD_eNKI_CamCAN_all_sampling_Q.csv")   # noqa

# %%

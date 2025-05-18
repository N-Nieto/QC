# %%
import sys
import json
import timeit
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))

from lib.data_processing import balance_data_age_gender_Qsampling, ConfoundRegressor_TIV  # noqa
from lib.data_loading import load_data_and_qc  # noqa
from lib.ml import classification_results_by_site  # noqa
from lib.utils import ensure_dir  # noqa

# Save Direction
save_dir = project_root / "output" / "ML" / "single_site"

# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10

save_dir = save_dir / ("N_bins_" + str(n_age_bins))
ensure_dir(save_dir)
# %%

experiment_description = """
Experiment description:
This script is used to train a logistic regression model to classify sex based on
brain imaging data. The data is split into training and testing sets using repeated stratified k-fold cross-validation. The model is trained on the training set and evaluated on the testing set, with results saved to a CSV file. The script also generates a JSON file containing experiment information.
Each site is sampled randomly: random_Q. The process is repeated to account for variability.
The script uses a confound regressor to regress out the effect of Total Intracranial Volume (TIV) from the data before training the model.
"""
# Select dataset
site_list = ("SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains")

# Age range
LOW_CUT_AGE = 18
HICH_CUT_AGE = 80
RANDOM_REPETITIONS = 20
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 23

clf = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
confound_regressor = ConfoundRegressor_TIV()
kf_out = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
)

# %%
results = []
sampling_list = ["random_Q"]
start_time = timeit.default_timer()
site_info = {}

for repeated in range(RANDOM_REPETITIONS):
    print("Repetition")
    print(repeated)
    site_info[repeated] = {}
    for row, site in enumerate(site_list):
        site_info[repeated][site] = {}
        for col, sampling in enumerate(sampling_list):
            print(site)
            print(sampling)
            # Load data and prepare it
            X, Y = load_data_and_qc(site=site)

            # This is the main function to obtain different cohorts from the data
            X, Y = balance_data_age_gender_Qsampling(
                X,
                Y,
                n_age_bins,
                sampling,
                low_cut_age=LOW_CUT_AGE,
                high_cut_age=HICH_CUT_AGE,
            )
            X = X.to_numpy()

            site_info[repeated][site][sampling] = {
                "num_samples": X.shape[0],
                "num_features": X.shape[1],
                "Q_mean": np.mean(Y["IQR"]).round(4).item(),
                "Q_std_dev": np.std(Y["IQR"]).round(4).item(),
                "Q_min_value": np.min(Y["IQR"]).round(4).item(),
                "Q_max_value": np.max(Y["IQR"]).round(4).item(),
            }

            Y["gender"] = Y["gender"].replace({"F": 0, "M": 1}).astype(int)
            Y["TIV"] = Y["TIV"].replace({np.nan: Y["TIV"].mean()})
            TIV = Y["TIV"].to_numpy()

            Y = Y["gender"].to_numpy()

            # Main loop
            for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):  # noqa
                print("FOLD: " + str(i_fold))

                # Patients used for train and internal XGB validation
                X_train = X[train_index, :].copy()
                Y_train = Y[train_index].copy()
                TIV_train = TIV[train_index]

                # Patients used to generete a prediction
                X_test = X[test_index, :].copy()
                Y_test = Y[test_index].copy()
                TIV_test = TIV[test_index]
                # regress out TIV
                confound_regressor.fit(X_train, Y_train, TIV_train)

                X_residual, Y_residual = confound_regressor.transform(
                    X_train, Y_train, TIV_train
                )

                X_test, Y_residual = confound_regressor.transform(
                    X_test, Y_test, TIV_test
                )

                # Fit model
                clf.fit(X_residual, Y_train)
                pred_test = clf.predict_proba(X_test)[:, 1]
                results = classification_results_by_site(
                    i_fold,
                    "Single site Test",
                    pred_test,
                    Y_test,
                    results,
                    sampling,
                    [site],
                )

                pred_train = clf.predict_proba(X_residual)[:, 1]
                results = classification_results_by_site(
                    i_fold,
                    "Single site Train",
                    pred_train,
                    Y_train,
                    results,
                    sampling,
                    [site],
                )

experiment_time = timeit.default_timer() - start_time
print("Time elapsed: " + str(experiment_time))
# create a DataFrame for easier handeling
results = pd.DataFrame(
    results,
    columns=[
        "Fold",
        "Model",
        "Balanced ACC",
        "AUC",
        "F1",
        "Recall",
        "QC_Sampling",
        "Site",
    ],
)
# %%
results.to_csv(
    save_dir
    / (
        "results_"
        + str(n_age_bins)
        + "_bins_single_site_random_sampling_Q"
        + str(RANDOM_REPETITIONS)
        + "_repetitions.csv"
    )
)
# %%
experiment_info = {
    "experiment_description": experiment_description,
    "experiment_time": experiment_time,
    "site_list": site_list,
    "sampling_list": sampling_list,
    "low_cut_age": LOW_CUT_AGE,
    "high_cut_age": HICH_CUT_AGE,
    "n_age_bins": n_age_bins,
    "n_splits": N_SPLITS,
    "n_repeats": N_REPEATS,
    "random_state": RANDOM_STATE,
    "sites_info": site_info,
    "classifier": clf.__class__.__name__,
    "confound_regressor": "TIV",
}
with open(
    save_dir
    / (
        "experiment_info_"
        + str(n_age_bins)
        + "_bins_single_site_random_sampling_Q"
        + str(RANDOM_REPETITIONS)
        + "_repetitions.json"
    ),
    "w",
) as f:
    json.dump(experiment_info, f, indent=4)

print("Experment done!")

# %%

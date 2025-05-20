# %%
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lib.data_loading import load_data_and_qc  # noqa
from lib.ml import compute_regression_results  # noqa
from lib.utils import ensure_dir  # noqa
from lib.data_processing import keep_desired_age_range  # noqa

# Save Direction
save_dir = project_root / "output" / "QC_prediction_from_features/"

ensure_dir(save_dir)
# %%
# Select dataset
site_list = ("SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains")

# Age range
low_cut_age = 18
high_cut_age = 80


# Age range
LOW_CUT_AGE = 18
HICH_CUT_AGE = 80

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 23

clf = LinearRegression()
kf_out = RepeatedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
)

results = []
sampling = "all_data_used"

y_true_loop = []
y_pred_loop = []
site_loop = []

for row, site in enumerate(site_list):
    print(site)
    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)

    # This is the main function to obtain different cohorts from the data
    Y = keep_desired_age_range(Y, LOW_CUT_AGE, HICH_CUT_AGE)
    Y = Y.loc[Y.index]
    X = X.loc[Y.index]

    Y["IQR"] = Y["IQR"].replace({np.nan: Y["IQR"].mean()})
    # filter the data
    Y = Y["IQR"].to_numpy()
    X = X.to_numpy()

    # Main loop
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):  # noqa
        print("FOLD: " + str(i_fold))

        # Patients used for train
        X_train = X[train_index, :]
        Y_train = Y[train_index]

        # Patients used to generete a prediction
        X_test = X[test_index, :]
        Y_test = Y[test_index]

        # Fit model
        clf.fit(X_train, Y_train)
        pred_test = clf.predict(X_test)
        results = compute_regression_results(
            i_fold, "QC Test", pred_test, Y_test, results, sampling, [site]
        )

        pred_train = clf.predict(X_train)
        results = compute_regression_results(
            i_fold, "QC Train", pred_train, Y_train, results, sampling, [site]
        )
        y_true_loop.append(Y_test)
        y_pred_loop.append(pred_test)
        site_loop.append([site] * len(pred_test))
# %%
# create a DataFrame for easier handeling
results = pd.DataFrame(
    results,
    columns={
        "Fold": str,
        "Model": str,
        "MAE": float,
        "R2": float,
        "QC_Sampling": list[str],
        "Site": str,
    },
)

results_loop = pd.DataFrame(
    {
        "y_true": np.concatenate(y_true_loop),
        "y_pred": np.concatenate(y_pred_loop),
        "Site": np.concatenate(site_loop),
    }
)
# %%
results.to_csv(str(save_dir / "results_QC_single_site.csv"))
results_loop.to_csv(str(save_dir / "results_QC_prediction_single_site.csv"))

# %%

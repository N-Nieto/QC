# %%
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from pathlib import Path

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))

from lib.data_loading import load_data_and_qc  # noqa
from lib.ml import compute_regression_results  # noqa
from lib.utils import ensure_dir  # noqa


# Save Direction relative to the project root
save_dir = project_root / Path("output/QC_prediction_from_features/")
# %%
ensure_dir(save_dir)
# %%
# Select dataset
site_list = ("SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains")

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10

clf = LinearRegression()
kf_out = RepeatedKFold(n_splits=5, n_repeats=1, random_state=23)

results = []
sampling = ["random_Q"]

y_true_loop = []
y_pred_loop = []
site_loop = []

X_pooled = pd.DataFrame()
Y_pooled = pd.DataFrame()
site_pooled = pd.DataFrame()
for row, site in enumerate(site_list):
    print(site)
    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)

    # This is the main function to obtain different cohorts from the data
    # X, Y = balance_data_age_gender_Qsampling(X, Y, n_age_bins, Q_sampling=sampling)

    Y["IQR"] = Y["IQR"].replace({np.nan: Y["IQR"].mean()})

    Y_site = Y["IQR"]
    site = pd.DataFrame([site] * len(Y_site))
    Y_site = pd.DataFrame(Y_site)
    X_site = X

    X_pooled = pd.concat([X_pooled, X_site])
    Y_pooled = pd.concat([Y_pooled, Y_site])
    site_pooled = pd.concat([site_pooled, site])
# %%
# Main loop
X = X_pooled.to_numpy()
Y = Y_pooled.to_numpy()
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):  # noqa
    print("FOLD: " + str(i_fold+1))

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
    site_loop.append(site_pooled.iloc[test_index].to_numpy())    
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
        "y_true": np.concatenate(y_true_loop).ravel(),
        "y_pred": np.concatenate(y_pred_loop).ravel(),
        "Site": np.concatenate(site_loop).ravel(),
    }
)
# %%
results.to_csv(str(save_dir / "results_QC_agregated_pooled_data.csv"))
results_loop.to_csv(str(save_dir / "results_QC_prediction_pooled_data.csv"))

# %%

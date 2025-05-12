# %%
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import permutation_test_score

from pathlib import Path

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))

from lib.data_loading import load_data_and_qc  # noqa
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


score_loop = []
permutation_loop = []
pvalue_loop = []
site_loop = []
for row, site in enumerate(site_list):
    print(site)
    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)

    # This is the main function to obtain different cohorts from the data

    Y["IQR"] = Y["IQR"].replace({np.nan: Y["IQR"].mean()})

    Y = Y["IQR"].to_numpy()
    X = X.to_numpy()
    score, permutation, pvalue = permutation_test_score(
        clf, X, Y, n_permutations=100, cv=kf_out, scoring="neg_mean_absolute_error",
     )
    print("Real score: ", score)
    print("Permutation score: ", permutation.mean())
    print("Permutation pvalue: ", pvalue)
    score_loop.append(score)
    permutation_loop.append(permutation.mean())
    pvalue_loop.append(pvalue)
    site_loop.append(site)

# %%
results_loop = pd.DataFrame(
    {
        "score": np.array(score_loop),
        "permutation_score": np.array(permutation_loop),
        "pvalue": np.array(pvalue_loop),
        "Site": np.array(site_loop),
    }
)
# %%
results_loop.to_csv(str(save_dir / "results_QC_prediction_single_site_permutation_test.csv"))

# %%

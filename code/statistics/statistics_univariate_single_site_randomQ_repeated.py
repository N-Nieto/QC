# %%
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind
from pathlib import Path

# To avoid the warning when converting the "F" and "M" to 0 and 1
pd.set_option("future.no_silent_downcasting", True)

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from lib.data_processing import balance_data_age_gender_Qsampling  # noqa
from lib.data_loading import load_data_and_qc  # noqa
from lib.data_processing import ConfoundRegressor_TIV  # noqa
from lib.utils import ensure_dir  # noqa

p_values = []  # To store p-values for each feature

save_dir = project_root / "output" / "statistics" / "single_site/"

n_age_bins = 3  # experiments were run using 10 or 3

save_dir = save_dir / ("N_bins_" + str(n_age_bins))
ensure_dir(save_dir)
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
random_q_repeated = 20

# randomly select participants
sampling = "random_Q"

confound_regressor = ConfoundRegressor_TIV()
p_values_df = pd.DataFrame()
for row, site in enumerate(site_list):
    print(site)
    # Load data and prepare it
    X_org, Y_org = load_data_and_qc(site=site)

    for repeated in range(random_q_repeated):
        print(repeated)

        # This is the main function to obtain different cohorts from the data
        X, Y = balance_data_age_gender_Qsampling(X_org, Y_org, n_age_bins, sampling)

        # Replace the gender to number
        Y.replace(
            {"gender": {"F": 0, "M": 1}, "TIV": {np.nan: Y["TIV"].mean()}}, inplace=True
        )
        TIV = Y["TIV"].astype(float).to_numpy()
        Y = Y["gender"].to_numpy()

        # Remove TIV as confound
        confound_regressor.fit(X.to_numpy(), Y, TIV)
        X_residual, Y_residual = confound_regressor.transform(X.to_numpy(), Y, TIV)

        X = pd.DataFrame(X_residual)
        p_values = []  # To store p-values for each feature
        t_stats = []
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
            t_stats.append(t_stat)
        # ----------------------------
        # 4. Convert p-values to a Pandas DataFrame for easy handling
        # ----------------------------

        p_values_df_loop = pd.DataFrame(
            {
                "Feature": X.columns,
                "P-value": p_values,
                "t-stat": t_stats,
                "sampling": sampling,
            }
        )
        p_values_df = pd.concat([p_values_df, p_values_df_loop])

    p_values_df.to_csv(
        save_dir
        / (
            "statistic_test_"
            + str(n_age_bins)
            + "bins_"
            + str(random_q_repeated)
            + "repeated_random_sampling_"
            + site
            + ".csv"
        )
    )

print("Experiment Done!")

# %%

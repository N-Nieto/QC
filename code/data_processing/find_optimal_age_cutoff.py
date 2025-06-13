# %%
import pandas as pd
from pathlib import Path
import sys
import numpy as np

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from lib.data_loading import load_data_and_qc  # noqa
from lib.utils import ensure_dir  # noqa
from lib.data_processing import (  # noqa
    balance_data_age_gender_Qsampling,
    get_age_bins,
)


site_list = [
    "AOMIC_ID1000",
    "AOMIC-PIOP1",
    "AOMIC-PIOP2",
    "AOMIC",
    "eNKI",
    "CamCAN",
    "SALD",
    "1000Brains",
    "GSP",
    "DLBS",
]


age_step = 1
age_bin_star = 3
age_bin_stop = 10
age_bins_to_test = np.linspace(
    start=age_bin_star,
    stop=age_bin_stop,
    num=(age_bin_stop - age_bin_star) + 1,
    dtype=int,
)

# %%
for site in site_list:
    save_dir = Path(project_root) / "lib" / "optimal_age_cuts" / site
    ensure_dir(save_dir)
    results_site = []

    X, Y = load_data_and_qc(site=site)
    Y["age"] = round(Y["age"])  # Ensure age is rounded to nearest integer
    N_original = X.shape[0]
    min_age = int(Y["age"].min())
    max_age = int(Y["age"].max())
    age_values = sorted(Y["age"].dropna().unique())

    for N_age_bins in age_bins_to_test:
        print(f"Finding optimal age cuts for {site} with N_bins = {N_age_bins}")

        # Load data and prepare it
        if len(age_values) < N_age_bins:
            N_bins = len(age_values)  # Adjust N_bins if not enough unique ages
            print(
                f"Not enough unique ages for for this age  bins {N_bins}, Finish experiment for site: {site}."
            )
            break
        else:
            N_bins = N_age_bins
        # Ensure we have enough unique ages to create the bins

        # Calculate minimum required age range
        min_required_range = (N_bins - 1) * age_step

        # Iterate through all possible low-high pairs with sufficient range
        for low in age_values:
            # Calculate minimum high age that satisfies bin requirements
            min_high = low + min_required_range

            # Only consider high ages that are at least min_high
            possible_highs = [
                age for age in age_values if age >= min_high and age >= low
            ]

            for high in possible_highs:
                current_age_diff = high - low

                X_low, Y_low = balance_data_age_gender_Qsampling(
                    X,
                    Y,
                    N_bins,
                    Q_sampling="low_q",
                    low_cut_age=low,
                    high_cut_age=high,
                )

                X_high, Y_high = balance_data_age_gender_Qsampling(
                    X,
                    Y,
                    N_bins,
                    Q_sampling="high_q",
                    low_cut_age=low,
                    high_cut_age=high,
                )

                current_age_diff = high - low

                current_N_low = X_low.shape[0]
                current_N_high = X_high.shape[0]

                assert current_N_low == current_N_high, "Sampling does not much"

                current_N = current_N_low

                age_bins_range = get_age_bins(Y, N_bins)

                # Determine what is the max number of images in the
                # formed age bins for each gender

                # Convert arrays to sets
                set1 = set(Y_low.index)
                set2 = set(Y_high.index)
                N_share = len(set1.intersection(set2))
                Share_percentage = current_N / N_share
                # Print the number interception participants
                results_site.append(
                    {
                        "site": site,
                        "Original_N": N_original,
                        "Obtained_N": current_N,
                        "N_share": N_share,
                        "n_images_each_bin": (current_N / 2) / N_bins,
                        "age_bins": N_bins,
                        "low_age_cut": low,
                        "high_age_cut": high,
                        "Share_percentage": Share_percentage,
                        "age_diff": current_age_diff,
                        "median_low": Y_low["IQR"].median(),
                        "median_hich": Y_high["IQR"].median(),
                        "mean_low": Y_low["IQR"].mean(),
                        "mean_hich": Y_high["IQR"].mean(),
                        "std_low": Y_low["IQR"].std(),
                        "std_high": Y_high["IQR"].std(),
                    }
                )
    # Save the results for each site
    results_site = pd.DataFrame(results_site)
    # Save to CSV if needed
    results_site.to_csv(
        save_dir / f"results_optimal_N_experiment_site_{site}.csv",
        index=False,
    )

print("Experiment done!")

# %%

# %%
import pandas as pd
from pathlib import Path
import sys
import numpy as np

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from lib.data_loading import load_data_and_qc  # noqa
from lib.utils import ensure_dir  # noqa
from lib.data_processing import balance_data_age_gender_Qsampling  # noqa


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

results_all = []

age_bins = [3, 10]

for N_bins_original in age_bins:
    for site in site_list:
        save_dir = (
            Path(project_root)
            / "lib"
            / "optimal_age_cuts"
            / f"N_bins_{N_bins_original}"
            / site
        )
        # ensure_dir(save_dir)
        results = []

        print(f"Finding optimal age cuts for {site} with N_bins = {N_bins_original}")
        # Load data and prepare it
        X, Y = load_data_and_qc(site=site)
        Y["age"] = round(Y["age"])  # Ensure age is rounded to nearest integer



        min_age = int(Y["age"].min())
        max_age = int(Y["age"].max())
        age_values = sorted(Y["age"].unique())

        if len(age_values) < N_bins_original:
            N_bins = len(age_values)  # Adjust N_bins if not enough unique ages
            print(
                f"Not enough unique ages for {site}. Adjusting N_bins to {N_bins}."
            )
        else:
            N_bins = N_bins_original
        # Ensure we have enough unique ages to create the bins

        # Calculate age_step as the minimum difference between consecutive ages
        if len(age_values) > 1:
            age_step = min(np.diff(age_values))
        else:
            age_step = 1  # default if only one age value exists

        best_N = 0
        best_low = min_age
        best_high = max_age
        best_age_diff = 0

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
                X_mock, Y_mock = balance_data_age_gender_Qsampling(
                    X,
                    Y,
                    N_bins,
                    Q_sampling="random_q",  # not important
                    low_cut_age=low,
                    high_cut_age=high,
                )
                current_age_diff = high - low

                current_N = X_mock.shape[0]
                results_all.append(
                    {
                        "site": site,
                        "low_age_cut": low,
                        "high_age_cut": high,
                        "Obtained_N": current_N,
                        "age_diff": current_age_diff,
                        "age_bins": N_bins,

                    }
                )
                # Check if this combination is better than our current best
                if (current_N > best_N) or (
                    current_N == best_N and current_age_diff > best_age_diff
                ):
                    best_N = current_N
                    best_low = low
                    best_high = high
                    best_age_diff = current_age_diff

        results.append(
            {
                "site": site,
                "low_age_cut": best_low,
                "high_age_cut": best_high,
                "optimal_N": best_N,
                "age_diff": best_age_diff,
                "N_bins": N_bins,
            }
        )
        # # Display results
        # print(results)

        results = pd.DataFrame(results)
        # Save to CSV if needed
        results.to_csv(
            save_dir / f"optimal_age_cuts_results_site_{site}_nbins_{N_bins_original}.csv",
            index=False,
        )
# %%
results = pd.DataFrame(results_all)
save_dir = (
    Path(project_root)
    / "lib"
    / "optimal_age_cuts")

results.to_csv(
    save_dir / "optimal_age_cuts_results_overall.csv",
    index=False,
)
print("Experiment done!")

# %%

import pandas as pd


def get_results_for_site_qc_sampling_age_bins(results_base, site, sampling, age_bins):
    results_dir = (
        results_base
        / "single_site"
        / "optimal_age_range"
        / ("N_bins_" + str(age_bins))
        / site
        / sampling
    )
    file_name = f"results_{age_bins}_bins_site_{site}_sampling_{sampling}.csv"
    results = pd.read_csv(results_dir / file_name, index_col=0)
    results = results[results["Model"] == "Single site Test"]
    return results

# %%
import sys
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)

from lib.data_processing import get_min_common_number_images_in_age_bins, filter_age_bins_with_qc # noqa
from lib.data_loading import load_data_and_qc                           # noqa
from lib.data_processing import keep_desired_age_range, get_age_bins          # noqa


site_list = ["SALD", "eNKI", "CamCAN"]

# Directions
data_dir = "/final_data_split/"
qc_dir = "/qc/"


# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 3


for row, site in enumerate(site_list):

    # Load data and prepare it
    X, Y = load_data_and_qc(site=site)
    print("wholedata " + site + str(Y.__len__()))
    Y = keep_desired_age_range(Y, low_cut_age, high_cut_age)

    age_bins = get_age_bins(Y, n_age_bins)

    # Determine what is the max number of images in the
    # formed age bins for each gender
    n_images = get_min_common_number_images_in_age_bins(Y, age_bins)
    # n_images = 10
    # get the images depending the QC
    index_low = filter_age_bins_with_qc(Y, age_bins,
                                        n_images, sampling="low_Q")
    print(site + str(index_low.__len__()))

    # get the images depending the QC
    index_high = filter_age_bins_with_qc(Y, age_bins,
                                         n_images, sampling="high_Q")

    # Convert arrays to sets
    set1 = set(index_low)
    set2 = set(index_high)

    # Create Venn diagram
    plt.figure(figsize=(8, 6))
    venn2([set1, set2], ('Low_QC', 'High_QC'))
    plt.title("Number of Shared participants across QC sampling strategies for " + str(site))       # noqa
    plt.show()
# %%

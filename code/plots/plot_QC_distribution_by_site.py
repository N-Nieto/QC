# %%
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sbn
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             # noqa
sys.path.append(project_root)

from lib.data_processing import balance_data_age_gender_Qsampling # noqa
from lib.data_loading import load_data_and_qc                           # noqa


# Directions
data_dir = "/final_data_split/"
qc_dir = "/qc/"
# %%
# Select dataset
site_list = ["SALD", "eNKI", "CamCAN"]
# site_list = ["SALD"]

# Age range
low_cut_age = 18
high_cut_age = 80
# Number of bins to split the age and keep the same number
# of images in each age bin
n_age_bins = 10


# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
# random dosen't care about QC
sampling_list = ["low_Q", "high_Q", "random_Q"]

i = 1
plt.figure(figsize=[15, 15])

for row, site in enumerate(site_list):
    for col, sampling in enumerate(sampling_list):

        print(site)
        print(sampling)
        # Load data and prepare it
        X, Y = load_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)

        # This is the main function to obtain different cohorts from the data
        X, Y = balance_data_age_gender_Qsampling(X, Y, n_age_bins, sampling)
        plt.subplot(3, 3, i)

        sbn.swarmplot(Y.IQR)
        plt.title("QC data for site: "+site)
        plt.ylabel("IQR")
        plt.xlabel(sampling)
        plt.ylim([1.5, 4.5])
        plt.grid()

        i = i+1

# %%

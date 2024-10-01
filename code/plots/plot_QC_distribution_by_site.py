# %%
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
dir_path = '../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.data_processing import get_min_common_number_images_in_age_bins, filter_age_bins_with_qc # noqa
from lib.data_loading import load_data_and_qc                           # noqa
from lib.data_processing import keep_desired_age_range, get_age_bins          # noqa
from lib.ml import results_to_df_qc_single_site, results_qc_single_site                  # noqa


# Directions
data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
save_dir = "/home/nnieto/Nico/Harmonization/QC/output/sex_classification/"
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

clf = LogisticRegression()

results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

# low_Q retains the images with HIGHER IQR
# high_Q retains the images with LOWER IQR
# random dosen't care about QC
sampling_list = ["low_Q", "high_Q", "random_Q"]
import matplotlib.pyplot as plt
import seaborn as sbn
plt.figure(figsize=[15, 15])
i=1
for row, site in enumerate(site_list):
    for col, sampling in enumerate(sampling_list):
        
        print(site)
        print(sampling)
        # Load data and prepare it
        X, Y = load_data_and_qc(data_dir=data_dir, qc_dir=qc_dir, site=site)

        Y = keep_desired_age_range(Y, low_cut_age, high_cut_age)

        age_bins = get_age_bins(Y, n_age_bins)

        # Determine what is the max number of images in the
        # formed age bins for each gender
        n_images = get_min_common_number_images_in_age_bins(Y, age_bins)

        # get the images depending the QC
        index = filter_age_bins_with_qc(Y, age_bins,
                                        n_images, sampling=sampling)
        
        # filter the data
        Y = Y.loc[index]
        X = X.loc[index]
        print(Y["gender"].nunique())

        plt.subplot(3, 3, i)

        sbn.swarmplot(Y.IQR)
        plt.title("QC data for site: "+site)
        plt.ylabel("IQR")
        plt.xlabel(sampling)
        plt.ylim([1.5, 4.5])
        plt.grid()

        i = i+1
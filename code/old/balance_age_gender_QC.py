# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import sys
from pathlib import Path

dir_path = '../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)

from lib.data_processing import get_min_common_number_images_in_age_bins, filter_age_bins_with_qc # noqa

sites = [["eNKI", 18, 80],
         ["CamCAN", 18, 80],
         ["SALD", 18, 80]]


root_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
save_dir = "/home/nnieto/Nico/Harmonization/data/qc/final_data_split/"
n_age_bins = 10
sampling = "low_Q"

# %%
# Main loop
for site, low_cut_age, high_cut_age in sites:

    X_data = pd.read_csv(root_dir + "X_" + site + ".csv")
    Y_data = pd.read_csv(root_dir + "Y_" + site + ".csv")
    qc_data = pd.read_csv(qc_dir+site+"_cat12.8.1_rois_thalamus.csv")
    qc_data.rename(columns={"SubjectID": "subject"}, inplace=True)

    # For the naming exeptions
    if site == "eNKI":
        qc_data = qc_data[qc_data["Session"] == "ses-BAS1"]
    if site == "SALD":
        qc_data['subject'] = qc_data['subject'].str.replace('sub-', '')
        qc_data['subject'] = pd.to_numeric(qc_data['subject'])

    Y_data = pd.merge(Y_data, qc_data[['subject', 'IQR']],
                      on='subject', how='left')

    # Remove under 18 patients
    idx_age = np.array(round(Y_data["age"]) >= low_cut_age)
    # Remove patients over the limit
    idx_age2 = np.array(high_cut_age >= round(Y_data["age"]))
    Y_data = Y_data[idx_age*idx_age2]

    # For getting the "age bins". We will have the same number
    # of images for each gender in each age bin
    age_min = round(Y_data["age"].min())
    age_max = round(Y_data["age"].max())
    steps = round((age_max-age_min) / n_age_bins)
    age_bins = range(age_min, age_max, steps)

    # Determine what is the max number of images in the
    # formed age bins for each gender
    n_images = get_min_common_number_images_in_age_bins(Y_data, age_bins)
    print(n_images)

    # get the images depending the QC
    index = filter_age_bins_with_qc(Y_data, age_bins,
                                    n_images, sampling=sampling)
    # filter the data
    Y_filt = Y_data.loc[index]
    X_filt = X_data.loc[index]
    plt.figure()
    sbn.swarmplot(qc_data.IQR)
    sbn.swarmplot(Y_filt.IQR)
    plt.legend(["All participants", "Selected participants"])
    plt.grid()
    plt.xlabel(site)
    plt.title("Selected patients with sampling method: " + sampling)
    print(site)
    print(Y_filt["gender"].value_counts().sum())
    print(Y_filt["gender"].value_counts())
    print("---------")
    print("Saving")
    Y_filt.to_csv(save_dir + "Y_" + site + "_" + sampling + ".csv")
    X_filt.to_csv(save_dir + "X_" + site + "_" + sampling + ".csv")
    print("Done")


# %%

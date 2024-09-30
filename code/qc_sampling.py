# %%
import pandas as pd
data_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
CamCAN = pd.read_csv(data_dir+"CamCAN_cat12.8.1_rois_thalamus.csv")
eNKI = pd.read_csv(data_dir+"eNKI_cat12.8.1_rois_thalamus.csv")
SALd = pd.read_csv(data_dir+"SALD_cat12.8.1_rois_thalamus.csv")

data_dir = "/home/nnieto/Nico/Harmonization/data/final_data_split/"
Y_eNKI = pd.read_csv(data_dir+"Y_eNKI.csv")
Y_CamCAN = pd.read_csv(data_dir+"Y_CamCAN.csv")
Y_SALD = pd.read_csv(data_dir+"Y_SALD.csv")
# %%

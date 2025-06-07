# %%
import pandas as pd
import numpy as np
site_list = ["SALD", "eNKI", "CamCAN", "AOMIC_ID1000", "1000Brains"]

data_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
save_dir = "/home/nnieto/Nico/Harmonization/QC/data/"
for site in site_list:
    X = pd.read_csv(data_dir + "X_"+site+".csv")
    X = X.dropna(axis=1)

    Y = pd.read_csv(data_dir + "Y_"+site+".csv")
    QC = pd.read_csv(data_dir + "QC_"+site+".csv")
    QC.rename(columns={"SubjectID": "subject"}, inplace=True)
    if site == "AOMIC_ID1000":
        QC["subject"] = QC["subject"]+"_site-ID1000"
        # For the naming exeptions
    if site == "eNKI":
        QC = QC[QC["Session"] == "ses-BAS1"]
    if site == "SALD":
        QC['subject'] = QC['subject'].str.replace('sub-', '')
        QC['subject'] = pd.to_numeric(QC['subject'])
    Y = pd.merge(Y, QC[['subject', 'IQR', "TIV"]],
                 on='subject', how='left')
    Y.drop(columns=["subject", "site"], inplace=True)

    shuffled_columns = np.random.permutation(X.columns)
    X = X[shuffled_columns]
    # Rename columns of X
    new_column_names = [i+1 for i in range(X.shape[1])]
    X.columns = new_column_names
    # 2. Generate a random permutation of row indices
    row_permutation = np.random.permutation(X.index)

    # Apply the row permutation to both X and Y
    X = X.loc[row_permutation].reset_index(drop=True)
    Y = Y.loc[row_permutation].reset_index(drop=True)

    print(X.shape)
    X.to_csv(save_dir + "X_"+site+".csv")
    Y.to_csv(save_dir + "Y_"+site+".csv")
# %%

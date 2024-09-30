import pandas as pd


def load_data_and_qc(data_dir, qc_dir, site):

    X_data = pd.read_csv(data_dir + "X_" + site + ".csv")
    X_data.dropna(axis=1, inplace=True)
    Y_data = pd.read_csv(data_dir + "Y_" + site + ".csv")
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

    return X_data, Y_data

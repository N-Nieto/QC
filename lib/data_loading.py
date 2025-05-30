import pandas as pd
import os


def load_data_and_qc(site):
    # Generate the path to the data.
    # The data must be stored in "data" inside the repo folder
    data_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/"
    )

    X_data = pd.read_csv(data_folder_path + "X_" + site + ".csv", index_col=0)
    Y_data = pd.read_csv(data_folder_path + "Y_" + site + ".csv", index_col=0)
    X_data = X_data.dropna(axis=1)
    return X_data, Y_data


def load_ROI_data_and_qc(data_dir, qc_dir, site):
    data = pd.read_csv(
        qc_dir + site + "_cat12.8.1_rois_Schaefer2018_400Parcels_17Networks_order.csv"
    )  # noqa

    # For the naming exeptions
    if site == "eNKI":
        data = data[data["Session"] == "ses-BAS1"]
        data.drop(columns="Session", inplace=True)

    if site == "SALD":
        data["SubjectID"] = data["SubjectID"].str.replace("sub-", "")
        data["SubjectID"] = pd.to_numeric(data["SubjectID"])

    X_data = data.drop(
        columns=["NCR", "ICR", "IQR", "GM", "WM", "CSF", "WMH", "TSA", "TIV"]
    )
    X_data = X_data.loc[:, ~X_data.columns.str.endswith("_WM")]
    X_data.dropna(axis=1, inplace=True)

    Y_data = pd.read_csv(data_dir + "Y_" + site + ".csv")
    Y_data.rename(columns={"subject": "SubjectID"}, inplace=True)
    qc_data = data.loc[:, ["SubjectID", "IQR", "TIV"]]
    Y_data = pd.merge(
        Y_data, qc_data[["SubjectID", "IQR", "TIV"]], on="SubjectID", how="left"
    )
    Y_data = Y_data[Y_data["SubjectID"].isin(X_data["SubjectID"])]
    X_data = X_data[X_data["SubjectID"].isin(Y_data["SubjectID"])]

    X_data.drop(columns="SubjectID", inplace=True)
    X_data.reset_index(inplace=True)
    Y_data.reset_index(inplace=True)

    return X_data, Y_data

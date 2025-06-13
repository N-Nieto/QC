import pandas as pd
from typing import Tuple
from pathlib import Path
from lib.data_processing import remove_extremely_low_and_missing_Q_samples


def load_data_and_qc(
    site: str,
    base_path: Path = Path().resolve().parents[1],
    threshold=4,
    QC_metric: str = "IQR",
    lower_better: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data for a given site.

    This function loads the X and Y data for the specified site from the "data" folder,
    removes columns with missing values from the X data, and returns the processed data.

    Parameters
    ----------
    site : str
        The site identifier used to locate the corresponding data files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the processed X data (features) and Y data (labels).
    """
    # Generate the path to the data.
    # The data must be stored in "data" inside the repo folder
    data_folder_path = base_path / "data/"

    if site == "Pooled":
        # For pooled data, load all sites' data
        NotImplementedError("Pooled for all ")
        # for s in site_list[:-1]:
        #     X_temp, Y_temp = load_data_and_qc(site=s)
        #     if "X" in locals():
        #         X_data = pd.concat([X_data, X_temp], ignore_index=True)
        #         Y_data = pd.concat([Y_data, Y_temp], ignore_index=True)
        #     else:
        #         X_data = X_temp
        #         Y_data = Y_temp
        X_data = pd.DataFrame()
        Y_data = pd.DataFrame()

    else:
        X_data = pd.read_csv(data_folder_path / ("X_" + site + ".csv"), index_col=0)
        Y_data = pd.read_csv(data_folder_path / ("Y_" + site + ".csv"), index_col=0)
        X_data = X_data.dropna(axis=1)

    X_data, Y_data = remove_extremely_low_and_missing_Q_samples(
        X=X_data,
        Y=Y_data,
        threshold=threshold,
        QC_metric=QC_metric,
        lower_better=lower_better,
    )

    return X_data, Y_data


def load_ROI_data_and_qc(
    data_dir: str, qc_dir: str, site: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess ROI data and quality control (QC) data for a given site.

    This function loads the ROI data and QC data for the specified site, applies site-specific
    preprocessing steps, merges the QC data with the Y data, and ensures consistency between
    the X and Y data. It also removes unnecessary columns and resets the indices.

    Parameters
    ----------
    data_dir : str
        The directory containing the Y data files.
    qc_dir : str
        The directory containing the QC data files.
    site : str
        The site identifier used to locate the corresponding data files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the processed X data (features) and Y data (labels).
    """
    data = pd.read_csv(
        qc_dir + site + "_cat12.8.1_rois_Schaefer2018_400Parcels_17Networks_order.csv"
    )

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

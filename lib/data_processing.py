import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


def balance_gender(data, min_images):
    male = data[data["gender"] == "M"]
    female = data[data["gender"] == "F"]

    male = male.sample(n=min_images, random_state=23)
    female = female.sample(n=min_images, random_state=23)

    data_balanced = pd.concat([male, female])

    return data_balanced


def retain_images(X, Y):
    return X.loc[Y.index, :]


def get_min_common_number_images_in_age_bins(Y_data, age_bins):
    min_image = 100000  # Initialize with a large number
    age_low = 0  # Initialize the lower bound of the first age bin
    for t, n in enumerate(age_bins):
        if t == 0:
            # Skip the first bin as it has no lower bound
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age * idx_age2]
            # replace the age for the next bin
            age_low = age_high
            # Get the minimun value for each bin
            min_image_new = Y_filt["gender"].value_counts().min()
            if min_image_new < min_image:
                min_image = min_image_new

    return min_image


def filter_age_bins_with_qc(
    Y_data: pd.DataFrame,
    age_bins,
    n_images: int,
    sampling: str,
    random_state=None,
    qc_metric: str = "IQR",
    lower_better: bool = True,
):
    bin = 0
    filter_index = pd.Index([])  # Initialize an empty index
    age_low = 0  # Initialize the lower bound of the first age bin
    for t, n in enumerate(age_bins):
        if t == 0:
            # Skip the first bin as it has no lower bound
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age * idx_age2]

            if sampling == "high_Q":
                # Sort by IQR in ascending order to get the lowest IQR values
                if lower_better:
                    ascending = True
                else:
                    ascending = False
                Y_filt = Y_filt.sort_values(by=qc_metric, ascending=ascending)
            elif sampling == "low_Q":
                if lower_better:
                    ascending = False
                else:
                    ascending = True
                # Sort by IQR in descending order to get the highest IQR values
                Y_filt = Y_filt.sort_values(by=qc_metric, ascending=ascending)
            else:
                # Random sampling
                Y_filt = Y_filt.sample(frac=1, random_state=random_state)

            # Sample n_images per gender
            males = Y_filt[Y_filt["gender"] == "M"].iloc[:n_images].index
            females = Y_filt[Y_filt["gender"] == "F"].iloc[:n_images].index

            if bin == 0:
                filter_index = males.append(females)
                bin = 1
            else:
                filter_index = filter_index.append(males)
                filter_index = filter_index.append(females)

            # Replace the age for the next bin
            age_low = age_high

    return filter_index


def keep_desired_age_range(
    Y: pd.DataFrame, low_cut_age: int, high_cut_age: int
) -> pd.DataFrame:
    """
    Filters the dataset to keep only rows where the age is within the specified range.

    Parameters
    ----------
    Y : pd.DataFrame
        DataFrame containing the data with an "age" column.
    low_cut_age : int
        The lower bound of the age range (inclusive).
    high_cut_age : int
        The upper bound of the age range (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows within the specified age range.
    """
    # Remove under low_cut_age patients
    idx_age = np.array(round(Y["age"]) >= low_cut_age)
    # Remove patients over the high_cut_age limit
    idx_age2 = np.array(high_cut_age >= round(Y["age"]))
    Y_sel = Y[idx_age * idx_age2]
    return Y_sel


def get_age_bins(Y: pd.DataFrame, n_age_bins: int) -> range:
    """
    Computes age bins for dividing the dataset into equal intervals.

    Parameters
    ----------
    Y : pd.DataFrame
        DataFrame containing the data with an "age" column.
    n_age_bins : int
        The number of age bins to create.

    Returns
    -------
    range
        A range object representing the age bins.
    """
    # For getting the "age bins". We will have the same number
    # of images for each gender in each age bin
    age_min = round(Y["age"].min())
    age_max = round(Y["age"].max())
    steps = round((age_max - age_min) / n_age_bins)
    age_bins = range(age_min, age_max, steps)
    return age_bins


def balance_data_age_gender_Qsampling(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_age_bins: int,
    Q_sampling: str,
    low_cut_age: int = 18,
    high_cut_age: int = 80,
    qc_metric: str = "IQR",
    lower_better: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Balances the dataset by age and gender while applying quality sampling.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the input features.
    Y : pd.DataFrame
        DataFrame containing the target variables, including "age" and "gender" columns.
    n_age_bins : int
        Number of age bins to divide the dataset into.
    Q_sampling : str
        Sampling strategy for quality control. Options are "high_Q", "low_Q", or "random".
    low_cut_age : int, optional
        The lower bound of the age range to keep (inclusive). Default is 18.
    high_cut_age : int, optional
        The upper bound of the age range to keep (inclusive). Default is 80.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the filtered input features (X) and target variables (Y).
    """
    Y = keep_desired_age_range(Y, low_cut_age, high_cut_age)

    age_bins = get_age_bins(Y, n_age_bins)

    # Determine the minimum number of images in the formed age bins for each gender
    n_images = get_min_common_number_images_in_age_bins(Y, age_bins)

    # Get the images based on the quality sampling strategy
    index = filter_age_bins_with_qc(
        Y,
        age_bins,
        n_images,
        sampling=Q_sampling,
        qc_metric=qc_metric,
        lower_better=lower_better,
    )

    # Filter the data
    Y = Y.loc[index]
    X = X.loc[index]
    return X, Y


class ConfoundRegressor_TIV(TransformerMixin, BaseEstimator):
    def __init__(self):
        # Initialize the linear regression models for each feature and Y
        self.feature_models = None
        self.y_model = None

    def fit(self, X, Y, TIV):
        """
        Fit the confound regression models using the training data.

        Parameters:
        X (np.ndarray): Input features with shape (n_samples, n_features)
        Y (np.ndarray): Target variable with shape (n_samples,)
        TIV (np.ndarray): Confound variable with shape (n_samples,)
        """
        # Ensure TIV is in the correct shape (n_samples, 1)
        TIV = TIV.reshape(-1, 1) if len(TIV.shape) == 1 else TIV

        # Initialize regression models for each feature and Y
        self.feature_models = []
        for i in range(X.shape[1]):
            reg = LinearRegression(n_jobs=-1).fit(TIV, X[:, i])
            self.feature_models.append(reg)

        # Fit the regression model for Y
        self.y_model = LinearRegression().fit(TIV, Y)

        return self

    def transform(self, X, Y, TIV):
        """
        Transform X and Y by removing the effect of the confound variable TIV.

        Parameters:
        X (np.ndarray): Input features with shape (n_samples, n_features)
        Y (np.ndarray): Target variable with shape (n_samples,)
        TIV (np.ndarray): Confound variable with shape (n_samples,)

        Returns:
        X_residual (np.ndarray): Features after removing the effect of TIV.
        Y_residual (np.ndarray): Target after removing the effect of TIV.
        """
        # Ensure TIV is in the correct shape (n_samples, 1)
        TIV = TIV.reshape(-1, 1) if len(TIV.shape) == 1 else TIV

        # Apply the confound regression to each feature in X
        X_residual = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_residual[:, i] = X[:, i] - self.feature_models[i].predict(TIV)

        # Apply the confound regression to Y
        Y_residual = Y - self.y_model.predict(TIV)

        return X_residual, Y_residual


def load_optimal_age_cuts(project_root, sites, n_age_bins):
    """
    Load the optimal age cuts from a csv file.
    """
    age_cutoffs = {}
    for site in sites:
        age_cutoffs[site] = {}
        site__optimal_age = pd.read_csv(
            project_root
            / "lib"
            / "optimal_age_cuts"
            / f"N_bins_{n_age_bins}"
            / site
            / f"optimal_age_cuts_results_site_{site}_nbins_{n_age_bins}.csv",
        )
        age_cutoffs[site]["low"] = site__optimal_age["low_age_cut"].values[0]
        age_cutoffs[site]["high"] = site__optimal_age["high_age_cut"].values[0]

    return age_cutoffs


def remove_extremely_low_and_missing_Q_samples(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    threshold: float = 4.0,
    QC_metric: str = "IQR",
    lower_better: bool = True,
):
    """
    Remove samples with IQR lower than a given threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix of shape (n_samples, n_features).
    Y : pandas.DataFrame
        DataFrame containing at least the column 'IQR' for each sample.
    threshold : float
        Minimum IQR value to retain a sample.

    Returns
    -------
    X_filtered : same type as X
        Feature matrix with low-IQR samples removed.
    Y_filtered : pandas.DataFrame
        DataFrame with low-IQR samples removed.
    """
    if QC_metric not in Y.columns:
        raise ValueError(f"QC metric '{QC_metric}' not found in Y DataFrame.")
    if lower_better:
        # If lower is better, we keep samples with IQR >= threshold
        index_low = threshold >= Y[QC_metric]
    else:
        # If higher is better, we keep samples with IQR <= threshold
        index_low = Y[QC_metric] >= threshold

    X = X[index_low]
    Y = Y[index_low]
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    return X, Y

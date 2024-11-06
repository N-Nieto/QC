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
    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            min_image = 1000
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age*idx_age2]
            # replace the age for the next bin
            age_low = age_high
            # Get the minimun value for each bin
            min_image_new = Y_filt["gender"].value_counts().min()
            if min_image_new < min_image:
                min_image = min_image_new

    return min_image


def filter_age_bins_with_qc(Y_data, age_bins, n_images, sampling,
                            random_state=None):
    bin = 0
    filter_index = pd.Index([])  # Initialize an empty index

    for t, n in enumerate(age_bins):
        if t == 0:
            age_low = n
            continue
        else:
            age_high = n
            # Filter images in the bin range
            idx_age = np.array(round(Y_data["age"]) >= age_low)
            idx_age2 = np.array(age_high >= round(Y_data["age"]))
            Y_filt = Y_data[idx_age * idx_age2]

            if sampling == 'high_Q':
                # Sort by IQR in ascending order to get the lowest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=True)
            elif sampling == 'low_Q':
                # Sort by IQR in descending order to get the highest IQR values
                Y_filt = Y_filt.sort_values(by='IQR', ascending=False)
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


def keep_desired_age_range(Y, low_cut_age, high_cut_age):
    # Remove under 18 patients
    idx_age = np.array(round(Y["age"]) >= low_cut_age)
    # Remove patients over the limit
    idx_age2 = np.array(high_cut_age >= round(Y["age"]))
    Y_sel = Y[idx_age*idx_age2]
    return Y_sel


def get_age_bins(Y, n_age_bins):
    # For getting the "age bins". We will have the same number
    # of images for each gender in each age bin
    age_min = round(Y["age"].min())
    age_max = round(Y["age"].max())
    steps = round((age_max-age_min) / n_age_bins)
    age_bins = range(age_min, age_max, steps)
    return age_bins


def balance_data_age_gender_Qsampling(X, Y, n_age_bins, Q_sampling,
                                      low_cut_age: int = 18,
                                      high_cut_age: int = 80):

    Y = keep_desired_age_range(Y, low_cut_age, high_cut_age)

    age_bins = get_age_bins(Y, n_age_bins)

    # Determine what is the max number of images in the
    # formed age bins for each gender
    n_images = get_min_common_number_images_in_age_bins(Y, age_bins)

    # get the images depending the QC
    index = filter_age_bins_with_qc(Y, age_bins,
                                    n_images, sampling=Q_sampling)

    # filter the data
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
            reg = LinearRegression().fit(TIV, X[:, i])
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

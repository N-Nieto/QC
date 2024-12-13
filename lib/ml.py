import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score
from typing import List, Union


def classification_results_by_site(i_fold: int, model: str,
                                   prob: np.ndarray,
                                   y: np.ndarray,
                                   result: List[List
                                                [Union[int, str, float]]],
                                   sampling: str,
                                   sites: List[str],
                                   repeated: int = None
                                   ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        result (List[List[Union[int, str, float]]]): List to store the results.
        sampling (str): Q Sampling scheme
        sites (List[int]):  List of sites
        repeated (int): Only used for randomly repeated

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # Calculate the predictions using the passed ths
    prediction = (prob > 0.5).astype(int)

    if sites.__len__() == 1:

        # Compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)
        recall = recall_score(y, prediction)
        # Append results
        if repeated is None:
            result.append([i_fold, model,
                           bacc, auc, f1,
                           recall,
                           sampling,
                           sites[0]
                           ])
        else:
            result.append([i_fold, model,
                           bacc, auc, f1,
                           recall,
                           sampling,
                           sites[0],
                           repeated
                           ])
    else:

        for site in pd.unique(sites):
            index = np.where(sites == site)[0]
            # Use the indices to select the corresponding values
            # from prediction, y, and prob
            prediction_site = prediction[index]
            y_site = y[index]
            prob_site = prob[index]
            # Compute all the metrics
            bacc = balanced_accuracy_score(y_site, prediction_site)
            auc = roc_auc_score(y_site, prob_site)
            f1 = f1_score(y_site, prediction_site)
            recall = recall_score(y_site, prediction_site)

            # Append results
            if repeated is None:
                result.append([i_fold, model,
                              bacc, auc, f1,
                              recall,
                              sampling,
                              site
                               ])
            else:
                result.append([i_fold, model,
                              bacc, auc, f1,
                              recall,
                              sampling,
                              site,
                              repeated
                               ])

    return result

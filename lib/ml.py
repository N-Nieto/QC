import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score


from typing import List, Union


def results_qc_single_site(i_fold: int, model: str,
                           prob: np.ndarray,
                           y: np.ndarray,
                           result: List[List[Union[int, str, float]]],
                           sampling: str,
                           site: str,
                           ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    # noqa
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # Calculate the predictions using the passed ths
    prediction = (prob > 0.5).astype(int)

    # Compute all the metrics
    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)
    recall = recall_score(y, prediction)

    # Append results
    result.append([i_fold, model,
                   bacc, auc, f1,
                   recall,
                   sampling,
                   site
                   ])

    return result


def results_to_df_qc_single_site(result: List[List[Union[int, str, float]]]
                                 ) -> pd.DataFrame:
    """
    Convert the list of results to a DataFrame.

    Parameters:
        result (List[List[Union[int, str, float]]]): List containing results.

    Returns:
        pd.DataFrame: DataFrame containing results with labeled columns.
    """
    result_df = pd.DataFrame(result,
                             columns=["Fold",
                                      "Model",
                                      "Balanced ACC",
                                      "AUC",
                                      "F1",
                                      "Recall",
                                      "QC_Sampling",
                                      "Site"
                                      ])
    return result_df

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from juharmonize import JuHarmonizeClassifier
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedStratifiedKFold

from typing import List, Union
import sys
from pathlib import Path
dir_path = '../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.data_processing import compute_classification_results          # noqa
from lib.data_processing import load_qc_dataset                         # noqa


# %%
save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/qc/"   # noqa
data_dir = "/home/nnieto/Nico/Harmonization/data/qc/final_data_split/"

qc_sampling = "high_Q"

X_SALD, Y_SALD = load_qc_dataset("SALD", qc_sampling, data_dir)
X_eNKI, Y_eNKI = load_qc_dataset("eNKI", qc_sampling, data_dir)
X_Camcan, Y_Camcan = load_qc_dataset("CamCAN", qc_sampling, data_dir)

Y = pd.concat([Y_SALD, Y_eNKI, Y_Camcan])

X = pd.concat([X_SALD, X_eNKI, X_Camcan])
X.dropna(axis=1, inplace=True)

# %%

Y["site"].replace({"SALD": 0, "eNKI": 1,
                   "CamCAN": 2}, inplace=True)
sites = Y["site"].reset_index()
Y["gender"].replace({"F": 0, "M": 1}, inplace=True)

Y = Y["gender"]
X = X.to_numpy()

# %%
clf = LogisticRegression()
JuHarmonize_model = JuHarmonizeClassifier(stack_model="logit",
                                          pred_model="logit")

print("Number of sites: " + str(sites.nunique()))
# %%
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])


harm_cheat, data_cheat_no_target = harmonizationLearn(data=X, # noqa
                                                      covars=covars)

covars['Target'] = Y.to_numpy().ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)
Y = Y.to_numpy()


# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]
    X_cheat_no_target_train = data_cheat_no_target[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]
    X_cheat_no_target_test = data_cheat_no_target[test_index, :]

    site_test = sites.iloc[test_index, :]

    Y_test = Y[test_index]

    # None model
    clf.fit(X_train, Y_train)
    pred_test = clf.predict_proba(X_test)[:, 1]
    results = compute_classification_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_train)[:, 1]
    results = compute_classification_results(i_fold, "None Train", pred_train, Y_train, results)                 # noqa

    # Cheat
    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_test)[:, 1]
    results = compute_classification_results(i_fold, "Cheat Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_cheat_train)[:, 1]
    results = compute_classification_results(i_fold, "Cheat Train", pred_train, Y_train, results)                 # noqa

    # Cheat no target
    clf.fit(X_cheat_no_target_train, Y_train)
    pred_test = clf.predict_proba(X_cheat_no_target_test)[:, 1]
    results = compute_classification_results(i_fold, "Cheat No Target Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(X_cheat_no_target_train)[:, 1]
    results = compute_classification_results(i_fold, "Cheat No Target Train", pred_train, Y_train, results)                 # noqa

    # # Leakage
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])
    covars_train['Target'] = Y_train.ravel()

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    covars_test['Target'] = Y_test.ravel()

    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "Leakage Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "Leakage Train", pred_train, Y_train, results)                 # noqa

    # No Target
    covars_train = pd.DataFrame(site_train["site"].to_numpy(),
                                columns=['SITE'])

    harm_model, harm_data = harmonizationLearn(X_train, covars_train)
    # Fit the model with the harmonizezd trian
    clf.fit(harm_data, Y_train)
    # covars
    covars_test = pd.DataFrame(site_test["site"].to_numpy(),
                               columns=['SITE'])
    harm_data_test = harmonizationApply(X_test,
                                        covars_test,
                                        harm_model)

    pred_test = clf.predict_proba(harm_data_test)[:, 1]
    results = compute_classification_results(i_fold, "No Target Test", pred_test, Y_test, results)                 # noqa

    pred_train = clf.predict_proba(harm_data)[:, 1]
    results = compute_classification_results(i_fold, "No Target Train", pred_train, Y_train, results)                 # noqa

    # # JuHarmonize
    JuHarmonize_model.fit(X=X_train, y=Y_train,
                          sites=site_train["site"].to_numpy())
    pred_test = JuHarmonize_model.predict_proba(X_test,
                                                sites=site_test["site"].to_numpy())[:, 1]           # noqa
    results = compute_classification_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                 # noqa

    pred_train = JuHarmonize_model.predict_proba(X_train,
                                                 sites=site_train["site"].to_numpy())[:, 1]         # noqa
    results = compute_classification_results(i_fold, "JuHarmonize Train", pred_train, Y_train, results)                 # noqa


# %%
def results_to_df(result: List[List[Union[int, str, float]]]) -> pd.DataFrame:
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
                                      ])
    return result_df


results = results_to_df(results)
# %%
# Save results
results.to_csv("/home/nnieto/Nico/Harmonization/QC/output/results_JuHarmonize.csv")   # noqa
# # %%
# import seaborn as sbn
# # data_random = results
# # site_order = ["Global", "eNKI", "CamCAN"]
# metric_to_plot = "AUC"
# import matplotlib.pyplot as plt
# # Plot
# pal = sbn.cubehelix_palette(5, rot=-.5, light=0.5, dark=0.2)
# _, ax = plt.subplots(1, 1, figsize=[12, 7])


# sbn.boxplot(
#     data=data_random, zorder=1,
#     x="Model", y=metric_to_plot, hue="Model",
#  dodge=False, ax=ax
# )

# plt.ylabel(metric_to_plot)
# plt.xlabel("Sites")
# plt.title("Gender Classification - Random Q")
# plt.grid(alpha=0.5, axis="y", c="black")
# plt.show()
# # %%
# import seaborn as sbn
# # data_low_q = results
# # site_order = ["Global", "eNKI", "CamCAN"]
# metric_to_plot = "AUC"
# import matplotlib.pyplot as plt
# # Plot
# pal = sbn.cubehelix_palette(5, rot=-.5, light=0.5, dark=0.2)
# _, ax = plt.subplots(1, 1, figsize=[12, 7])


# sbn.boxplot(
#     data=data_low_q, zorder=1,
#     x="Model", y=metric_to_plot, hue="Model",
#  dodge=False, ax=ax
# )

# plt.ylabel(metric_to_plot)
# plt.xlabel("Sites")
# plt.title("Gender Classification - Low Q")
# plt.grid(alpha=0.5, axis="y", c="black")
# plt.show()
# # %%
# import seaborn as sbn
# data_high_q = results
# # site_order = ["Global", "eNKI", "CamCAN"]
# metric_to_plot = "AUC"
# import matplotlib.pyplot as plt
# # Plot
# pal = sbn.cubehelix_palette(5, rot=-.5, light=0.5, dark=0.2)
# _, ax = plt.subplots(1, 1, figsize=[12, 7])


# sbn.boxplot(
#     data=data_high_q, zorder=1,
#     x="Model", y=metric_to_plot, hue="Model",
#  dodge=False, ax=ax
# )

# plt.ylabel(metric_to_plot)
# plt.xlabel("Sites")
# plt.title("Gender Classification - High Q")
# plt.grid(alpha=0.5, axis="y", c="black")
# plt.show()
# # %%
# data_high_q.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/qc/sex_classification_independant_high_qc.csv")
# data_low_q.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/qc/sex_classification_independant_low_qc.csv")
# data_random.to_csv("/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/qc/sex_classification_independant_random_qc.csv")
# # %%
# print(928+818+651+1144+494)


# # %%

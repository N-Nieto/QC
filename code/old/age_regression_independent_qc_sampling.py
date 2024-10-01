# %%
import pandas as pd
from juharmonize import JuHarmonizeRegressor
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.model_selection import RepeatedKFold
from skrvm import RVR
import sys
from pathlib import Path
dir_path = '../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)
from lib.data_processing import compute_classification_results  # noqa
from lib.data_processing import load_qc_dataset                 # noqa

# %%
save_dir = "/home/nnieto/Nico/Harmonization/harmonize_project/scratch/output/qc/"   # noqa
data_dir = "/home/nnieto/Nico/Harmonization/data/qc/final_data_split/"

qc_sampling = "random"

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

Y = Y["age"].to_numpy().astype(int)
X = X.to_numpy()

# %%
results = []

kf_out = RepeatedKFold(n_splits=5,
                       n_repeats=1,
                       random_state=23)

covars = pd.DataFrame(sites["site"].to_numpy(), columns=['SITE'])

covars['Target'] = Y.ravel()

harm_cheat, data_cheat = harmonizationLearn(data=X, # noqa
                                            covars=covars)
data_cheat = pd.DataFrame(data_cheat)


# %%
stack_model = RVR(kernel="poly", degree=1)
pred_model = RVR(kernel="poly", degree=1)

JuHarmonize_model = JuHarmonizeRegressor(stack_model=stack_model,
                                         pred_model=pred_model)


y_true_loop = []
sites_loop = []
pred_none = []
pred_cheat = []
pred_notarget = []
pred_leak = []
pred_juharmonize = []

# %%
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X)):       # noqa
    print("FOLD: " + str(i_fold))
    # Train
    X_train = X[train_index, :]
    X_cheat_train = data_cheat.iloc[train_index, :]

    site_train = sites.iloc[train_index, :]
    Y_train = Y[train_index]

    # Test
    X_test = X[test_index, :]
    X_cheat_test = data_cheat.iloc[test_index, :]

    site_test = sites.iloc[test_index, :]
    sites_loop.append(site_test["site"].to_numpy())
    Y_test = Y[test_index]
    y_true_loop.append(Y_test)

    # Models
    # None model
    print("None")
    clf = RVR(kernel="poly", degree=1)

    clf.fit(X_train, Y_train)
    pred_test = clf.predict(X_test)
    results = compute_classification_results(i_fold, "None Test", pred_test, Y_test, results)                 # noqa
    pred_none.append(pred_test)
    print("Cheat")

    # Cheat
    clf = RVR(kernel="poly", degree=1)

    clf.fit(X_cheat_train, Y_train)
    pred_test = clf.predict(X_cheat_test)
    results = compute_classification_results(i_fold, "Cheat Test", pred_test, Y_test, results)                 # noqa
    pred_cheat.append(pred_test)

    print("Leakage")
    # # Leakage
    clf = RVR(kernel="poly", degree=1)

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

    pred_test = clf.predict(harm_data_test)
    results = compute_classification_results(i_fold, "Leakage Test", pred_test, Y_test, results)                       # noqa
    pred_leak.append(pred_test)

    print("No Target")

    # No Target
    clf = RVR(kernel="poly", degree=1)

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

    pred_test = clf.predict(harm_data_test)
    results = compute_classification_results(i_fold, "No Target Test", pred_test, Y_test, results)                     # noqa
    pred_notarget.append(pred_test)

    print("JuHarmonize")
    # # JuHarmonize
    JuHarmonize_model.fit(X=X_train, y=Y_train,
                          sites=site_train["site"].to_numpy())
    pred_test = JuHarmonize_model.predict(X_test,
                                        sites=site_test["site"].to_numpy())                     # noqa
    results = compute_classification_results(i_fold, "JuHarmonize Test", pred_test, Y_test, results)                   # noqa
    pred_juharmonize.append(pred_test)

# %%
print("Saving")
pd.DataFrame(y_true_loop).to_csv(save_dir+qc_sampling+"_y_true.csv")
pd.DataFrame(sites_loop).to_csv(save_dir+qc_sampling+"_sites.csv")

pd.DataFrame(pred_none).to_csv(save_dir+qc_sampling+"_y_pred_none.csv")
pd.DataFrame(pred_cheat).to_csv(save_dir+qc_sampling+"_y_pred_cheat.csv")
pd.DataFrame(pred_notarget).to_csv(save_dir+qc_sampling+"_y_pred_notarget.csv")
pd.DataFrame(pred_leak).to_csv(save_dir+qc_sampling+"_y_pred_leak.csv")
pd.DataFrame(pred_juharmonize).to_csv(save_dir+qc_sampling+"_y_pred_juharmonize.csv")           # noqa

pd.DataFrame(results, columns=["Fold", "Harmonization Scheme", "MAE", "R2",
                               "Age bias"]
             ).to_csv(save_dir+"results_age_regression_dependant_"+qc_sampling+".csv")       # noqa
# %%

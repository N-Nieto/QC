# %%
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
dir_path = '../../lib/'
__file__ = dir_path + "data_processing.py"
to_append = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(to_append)


from lib.ml import results_to_df_qc_single_site, results_qc_single_site                  # noqa

# %%
qc_dir = "/home/nnieto/Nico/Harmonization/data/qc/"
site = "AOMIC_ID1000"
runs = [1, 2, 3]
data_dir = qc_dir
# Load data
Y_ID1000 = pd.read_csv(data_dir+"Y_ID1000.csv")
# Put all names in the same format
Y_ID1000['subject'] = Y_ID1000['subject'].str.replace(r'_site-ID\d+', '', regex=True)
# Put the subject name as in the QC data
Y_ID1000.rename(columns={"subject": "SubjectID"}, inplace=True)

# Load QC data
for run_number in runs:
    qc_data = pd.read_csv(qc_dir+site+"_cat12.8.1_run-"+str(run_number)+"_rois_thalamus.csv")
    qc_data = qc_data.loc[:,["SubjectID", "IQR"]]
    qc_data.rename(columns={"IQR": "IQR_run"+str(run_number)}, inplace=True)
    Y_ID1000 = pd.merge(Y_ID1000, qc_data, on='SubjectID', how='left')

# Remove those subject without QC data
Y_ID1000 = Y_ID1000.dropna()
# %%
data =Y_ID1000
 # Identify the columns where the minimum and maximum IQR values occur for each patient
data['min_IQR_run'] = data[['IQR_run1', 'IQR_run2', 'IQR_run3']].idxmin(axis=1)
data['max_IQR_run'] = data[['IQR_run1', 'IQR_run2', 'IQR_run3']].idxmax(axis=1)

data['min_IQR'] = data[['IQR_run1', 'IQR_run2', 'IQR_run3']].min(axis=1)
data['max_IQR'] = data[['IQR_run1', 'IQR_run2', 'IQR_run3']].max(axis=1)


# Calculate the absolute difference between the highest and lowest IQR
data['IQR_diff'] = data[['IQR_run1', 'IQR_run2', 'IQR_run3']].max(axis=1) - data[['IQR_run1', 'IQR_run2', 'IQR_run3']].min(axis=1)


def retain_patients_above_qc_difference(data, thd):

    selected_data = data[data["IQR_diff"] >= thd]
    return selected_data


# data = retain_patients_above_qc_difference(data, 0.1)
X_run1 = pd.read_csv(data_dir + "AOMIC_ID1000_cat12.8.1_run-1_rois_thalamus.csv")
X_run2 = pd.read_csv(data_dir + "AOMIC_ID1000_cat12.8.1_run-2_rois_thalamus.csv")
X_run3 = pd.read_csv(data_dir + "AOMIC_ID1000_cat12.8.1_run-3_rois_thalamus.csv")


to_keep = ['SubjectID', 'min_IQR_run', 'max_IQR_run',
           "min_IQR", "max_IQR", "IQR_diff", "gender"]
# Merge the IQR data with each X_run dataframe on 'Subject_ID' to align indices
X_run1 = X_run1.merge(data[to_keep], on='SubjectID')
X_run2 = X_run2.merge(data[to_keep], on='SubjectID')
X_run3 = X_run3.merge(data[to_keep], on='SubjectID')
# %%
# Select rows for X_high_IQR based on max_IQR_run
high_IQR_data = pd.concat([
    X_run1[X_run1['max_IQR_run'] == 'IQR_run1'],
    X_run2[X_run2['max_IQR_run'] == 'IQR_run2'],
    X_run3[X_run3['max_IQR_run'] == 'IQR_run3']
])

# Select rows for X_low_IQR based on min_IQR_run
low_IQR_data = pd.concat([
    X_run1[X_run1['min_IQR_run'] == 'IQR_run1'],
    X_run2[X_run2['min_IQR_run'] == 'IQR_run2'],
    X_run3[X_run3['min_IQR_run'] == 'IQR_run3']
])

print(low_IQR_data.shape)

low_IQR_data["IQR"] = low_IQR_data["min_IQR"]
high_IQR_data["IQR"] = high_IQR_data["max_IQR"]

low_IQR_data = low_IQR_data.sort_values(by='IQR_diff', ascending=False)

high_IQR_data = high_IQR_data.sort_values(by='IQR_diff', ascending=False)

# %%
features = ['lAnterior',
            'lCentral_Lateral-Lateral_Posterior-Medial_Pulvinar',
            'lMedio_Dorsal',
            'lPulvinar', 'lVentral_Anterior', 'lVentral_Latero_Dorsal',
            'lVentral_Latero_Ventral', 'rAnterior',
            'rCentral_Lateral-Lateral_Posterior-Medial_Pulvinar',
            'rMedio_Dorsal',
            'rPulvinar', 'rVentral_Anterior', 'rVentral_Latero_Dorsal',
            'rVentral_Latero_Ventral']

high_IQR_data["gender"].replace({"F": 0, "M": 1}, inplace=True)
low_IQR_data["gender"].replace({"F": 0, "M": 1}, inplace=True)
Y = high_IQR_data["gender"].to_numpy()
Y2 = low_IQR_data["gender"].to_numpy()
X_high_IQR = high_IQR_data.loc[:, features].to_numpy()
X_low_IQR = low_IQR_data.loc[:, features].to_numpy()
# %%

clf = LogisticRegression()
results = []

kf_out = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=5,
                                 random_state=23)

predictions_train_good_test_good = {}
predictions_train_good_test_bad = {}
predictions_train_bad_test_good = {}
predictions_train_bad_test_bad = {}

y_true_loop = {}
# Main loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X_high_IQR, y=Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_bad = X_high_IQR[train_index, :]
    X_train_good = X_low_IQR[train_index, :]

    Y_train = Y[train_index]

    # Patients used to generete a prediction
    X_test_bad = X_high_IQR[test_index, :]
    X_test_good = X_low_IQR[test_index, :]

    Y_test = Y[test_index]
    y_true_loop[str(i_fold)] = list(Y_test)
    # None model
    clf.fit(X_train_bad, Y_train)
    pred_test = clf.predict_proba(X_test_bad)[:, 1]
    predictions_train_bad_test_bad[str(i_fold)] = list(pred_test)
    results = results_qc_single_site(i_fold, "Train high IQR (WORST DATA) - Test high IQR (WORST DATA)", pred_test, Y_test, results, "all", "AOMIC")                 # noqa
    pred_test = clf.predict_proba(X_test_good)[:, 1]
    predictions_train_bad_test_good[str(i_fold)] = list(pred_test)

    results = results_qc_single_site(i_fold, "Train high IQR (WORST DATA) - Test low IQR (BEST DATA)", pred_test, Y_test, results, "all", "AOMIC")                 # noqa

    # None model
    clf.fit(X_train_good, Y_train)
    pred_test = clf.predict_proba(X_test_bad)[:, 1]
    predictions_train_good_test_bad[str(i_fold)] = list(pred_test)
    results = results_qc_single_site(i_fold, "Train Low IQR (BEST DATA) - Test high IQR (WORST DATA)", pred_test, Y_test, results, "all", "AOMIC")                 # noqa
    pred_test = clf.predict_proba(X_test_good)[:, 1]
    predictions_train_good_test_good[str(i_fold)] = list(pred_test)
    results = results_qc_single_site(i_fold, "Train Low IQR (BEST DATA) - Test low IQR (BEST DATA)", pred_test, Y_test, results, "all", "AOMIC")                 # noqa


results = results_to_df_qc_single_site(results)
# %%

predictions_train_good_test_good = pd.DataFrame.from_dict(predictions_train_good_test_good, orient='index')
predictions_train_good_test_bad = pd.DataFrame.from_dict(predictions_train_good_test_bad, orient='index')
predictions_train_bad_test_good = pd.DataFrame.from_dict(predictions_train_bad_test_good, orient='index')
predictions_train_bad_test_bad = pd.DataFrame.from_dict(predictions_train_bad_test_bad, orient='index')

y_true_loop = pd.DataFrame.from_dict(y_true_loop, orient='index')
# %%

min_iqr = 0
max_iqr = 1

# sbn.lineplot(x=[min_iqr,max_iqr],

#              y=[min_iqr,max_iqr])
plt.grid()

sbn.swarmplot(y=predictions_train_good_test_good.iloc[0,:]- predictions_train_good_test_bad.iloc[0,:],
                x=y_true_loop.iloc[0,:])
# %%
sbn.lineplot(x=[min_iqr,max_iqr],
             y=[min_iqr,max_iqr])
plt.grid()
sbn.scatterplot(x=predictions_train_bad_test_good.iloc[0,:],
                y=predictions_train_bad_test_bad.iloc[0,:],
                hue=y_true_loop.iloc[0,:])
# for index in predictions_train_good_test_good.index:
#     sbn.scatterplot(x=predictions_train_good_test_good.loc[index,:],
#                     y=predictions_train_good_test_bad.loc[index,:],
#                     hue=y_true_loop.loc[index,:])
# %%
order = ["Train Low IQR (BEST DATA) - Test low IQR (BEST DATA)",
         "Train Low IQR (BEST DATA) - Test high IQR (WORST DATA)",
         "Train high IQR (WORST DATA) - Test low IQR (BEST DATA)",
         "Train high IQR (WORST DATA) - Test high IQR (WORST DATA)"]
plt.figure(figsize=[20, 10])
sbn.boxplot(data=results, x="Model", y="AUC", order=order)

plt.grid()
# %%

X_diff = pd.DataFrame(X_high_IQR - X_low_IQR, columns=features)
# %%
plt.figure(figsize=[20,10])
sbn.scatterplot(y = X_diff.mean(axis=1), x=data["IQR_diff"])
plt.ylabel("Overall feature difference")
plt.grid()

# %%

plt.figure(figsize=[20,10])
sbn.scatterplot(y = X_high_IQR.std(axis=1), x=data["IQR_diff"])


sbn.scatterplot(y = X_low_IQR.std(axis=1), x=data["IQR_diff"])
plt.ylabel("Overall feature difference")
plt.grid()

# %%
plt.figure(figsize=[20,10])
sbn.scatterplot(x = low_IQR_data["IQR"], y=high_IQR_data["IQR"])

min_iqr = 1.85
max_iqr = 2.8

sbn.lineplot(x=[min_iqr,max_iqr],
             y=[min_iqr,max_iqr])

plt.ylabel("IQR (Worst)")
plt.xlabel("IQR (Best)")

plt.grid()
# %%

plt.figure(figsize=[20,10])
sbn.scatterplot(x = low_IQR_data["IQR"], y=high_IQR_data["IQR_diff"])

min_iqr = 1.75
max_iqr = 3


plt.ylabel("IQR addition (Makes it worst)")
plt.xlabel("IQR (Best)")

plt.grid()
# %%
data2 = retain_patients_above_qc_difference(data, 0.1)

data2['min_IQR'] = data2[['IQR_run1', 'IQR_run2', 'IQR_run3']].min(axis=1)
data2['max_IQR'] = data2[['IQR_run1', 'IQR_run2', 'IQR_run3']].max(axis=1)

# plt.figure(figsize=[20,10])
sbn.scatterplot(y = data2["min_IQR"], x=data2["max_IQR"])


plt.ylabel("Min IQR")
plt.xlabel("Max IQR")

plt.grid()


# %%

metric = np.std(X_high_IQR - X_low_IQR, axis=0) / np.mean(X_high_IQR - X_low_IQR, axis=0) 
plt.plot(metric)
plt.xlabel("Features ")
plt.ylabel("Std(features diff) / mean(features diff)")
plt.grid()

# %%
plt.figure(figsize=[20, 10])
sbn.scatterplot(y=X_high_IQR["IQR"], x=X_low_IQR["IQR"])


plt.ylabel("Best QC for run")
plt.grid()

# %%

final_dataset = pd.DataFrame()
while final_dataset.__len__() != high_IQR_data["SubjectID"].nunique():
    
    final_dataset = pd.concat([final_dataset, low_IQR_data.iloc[0,:]])
    high_IQR_data = high_IQR_data[high_IQR_data["SubjectID"] != final_dataset["SubjectID"]]
    final_dataset = pd.concat([final_dataset, high_IQR_data.iloc[0,:]])
    high_IQR_data = low_IQR_data[low_IQR_data["SubjectID"] != final_dataset["SubjectID"]]

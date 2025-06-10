# %%
import pandas as pd
from pathlib import Path


# Define the paths to the datasets
base_dir = Path('/home/nnieto/Nico/QC_project/QC/data')
datasets = ['AOMIC_ID1000', 'AOMIC-PIOP1', 'AOMIC-PIOP2']

X_list = []
Y_list = []

for ds in datasets:
    ds_path = base_dir
    X_file = ds_path / f'X_{ds}.csv'
    Y_file = ds_path / f'Y_{ds}.csv'
    if X_file.exists() and Y_file.exists():
        X_list.append(pd.read_csv(X_file))
        Y_list.append(pd.read_csv(Y_file))

    else:
        print(f"Warning: Missing X or Y in {ds_path}")

# Concatenate all dataframes
X_merged = pd.concat(X_list, ignore_index=True)
Y_merged = pd.concat(Y_list, ignore_index=True)

X_merged.to_csv(base_dir / 'X_AOMIC.csv', index=False)
Y_merged.to_csv(base_dir / 'Y_AOMIC.csv', index=False)

# %%

# %%
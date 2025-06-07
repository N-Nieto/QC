# %%
#
import sys
import pandas as pd
from pathlib import Path

project_root = Path().resolve().parents[2]
datalad_dir = project_root / "QC_ML_features" / "outputs/"

sys.path.append(str(project_root / "QC"))

from lib.datalad_data_merge import (  # noqa
    load_combine_split_site_data,
    load_patients_tsv,
    processing_participant_tsv,
    load_IQR_TIV,
    load_X,
)

# General
processing_pipeline = "_cat12.8.1"
save_dir = project_root / "QC" / "data"
save_flag = False
# %%

site = "DLBS"
# %%
participants = load_patients_tsv(datalad_dir, site, processing_pipeline)
# %%
participants = processing_participant_tsv(participants, site)
# %%
IQR = load_IQR_TIV(datalad_dir, site, processing_pipeline)
# %%

x = load_X(datalad_dir, site, processing_pipeline, s=4, r=8)
# %%
merged = pd.merge(left=IQR, right=x, on="participant_id")
merged = pd.merge(left=merged, right=participants, on="participant_id")

y = merged.loc[:, ["age", "gender", "IQR", "TIV"]]
y["gender"] = y["gender"].str.upper()
y["age"] = round(y["age"])
y.replace(to_replace={"gender": {"FEMALE": "F", "MALE": "M"}})

X = merged.drop(columns=["participant_id", "age", "gender", "IQR", "TIV"])

# %%
site = "SALD"
X, y = load_combine_split_site_data(datalad_dir, site, processing_pipeline)
if save_flag:
    y.to_csv(save_dir / f"Y_{site}.csv")
    X.to_csv(save_dir / f"X_{site}.csv")

# %%
site = "AOMIC_ID1000"
X, y = load_combine_split_site_data(datalad_dir, site, processing_pipeline)

if save_flag:
    y.to_csv(save_dir / f"Y_{site}.csv")
    X.to_csv(save_dir / f"X_{site}.csv")
# %%

# %%
# 1000brains
site = "1000brains"
X, y = load_combine_split_site_data(datalad_dir, site, processing_pipeline)

if save_flag:
    y.to_csv(save_dir / f"Y_{site}.csv")
    X.to_csv(save_dir / f"X_{site}.csv")

# %%
# GSP
site = "GSP"

X, y = load_combine_split_site_data(datalad_dir, site, processing_pipeline)

if save_flag:
    y.to_csv(save_dir / f"Y_{site}.csv")
    X.to_csv(save_dir / f"X_{site}.csv")


# %%
save_flag =  True
site = "DLBS"

X, y = load_combine_split_site_data(datalad_dir, site, processing_pipeline)
if save_flag:
    y.to_csv(save_dir / f"Y_{site}.csv")
    X.to_csv(save_dir / f"X_{site}.csv")
# %%

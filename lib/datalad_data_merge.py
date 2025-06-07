import pandas as pd


def load_combine_split_site_data(datalad_dir, site, processing_pipeline):
    participants = load_patients_tsv(datalad_dir, site, processing_pipeline)
    participants = processing_participant_tsv(participants, site)

    IQR = load_IQR_TIV(datalad_dir, site, processing_pipeline)
    x = load_X(datalad_dir, site, processing_pipeline, s=4, r=8)

    merged = pd.merge(left=IQR, right=x, on="participant_id")
    merged = pd.merge(left=merged, right=participants, on="participant_id")

    y = merged.loc[:, ["age", "gender", "IQR", "TIV"]]
    y["gender"] = y["gender"].str.upper()
    y["age"] = round(y["age"])
    y.replace(to_replace={"gender": {"FEMALE": "F", "MALE": "M"}})

    X = merged.drop(columns=["participant_id", "age", "gender", "IQR", "TIV"])

    return X, y


def load_patients_tsv(datalad_dir, site, processing_pipeline):
    participants = pd.read_csv(
        datalad_dir / (site + processing_pipeline) / "participants.tsv",
        sep="\t",
        header=0,
    )
    return participants


def load_IQR_TIV(datalad_dir, site, processing_pipeline):
    if site == "AOMIC_ID1000":
        IQR = pd.read_csv(
            datalad_dir
            / (site + processing_pipeline)
            / f"{site + processing_pipeline}_run-1_rois_thalamus.csv",
            usecols=["SubjectID", "IQR", "TIV"],
        )
        IQR.rename(columns={"SubjectID": "participant_id"}, inplace=True)

    elif site == "eNKI":
        IQR = pd.read_csv(
            datalad_dir
            / (site + processing_pipeline)
            / f"{site + processing_pipeline}_rois_thalamus.csv",
            usecols=["SubjectID", "IQR", "TIV", "Session"],
        )
        IQR.rename(columns={"SubjectID": "participant_id"}, inplace=True)
        # IQR.query("Session == 'ses-CLGA'", inplace=True)

    elif "SALD":
        IQR = pd.read_csv(
            datalad_dir
            / (site + processing_pipeline)
            / f"{site + processing_pipeline}_rois_thalamus.csv",
            usecols=["SubjectID", "IQR", "TIV"],
        )

        IQR.rename(columns={"SubjectID": "participant_id"}, inplace=True)

    elif site == "GSP" or site == "1000brains":
        IQR = pd.read_csv(
            datalad_dir
            / (site + processing_pipeline)
            / f"{site + processing_pipeline}_rois_thalamus.csv",
            usecols=["SubjectID", "IQR", "TIVSession"],
        )

        IQR.rename(columns={"SubjectID": "participant_id"}, inplace=True)

        IQR.query("Session == 'ses-1'", inplace=True)
        IQR.drop(columns=["Session"], inplace=True)
    else:
        ValueError(f"Invalid site name: {site}")

    return IQR


def load_X(datalad_dir, site, processing_pipeline, s: int, r: int):
    x = pd.read_csv(
        datalad_dir
        / (site + processing_pipeline)
        / f"{site + processing_pipeline}_p1_r{r}s{s}.csv",
        index_col=0,
    )
    x.drop(columns=["subj_path"], inplace=True)

    if site == "SALD":
        x["participant_id"] = x["subject"].str.split("_T1w").str[0]
        x["participant_id"] = "sub-" + x["participant_id"]
        x.drop(columns=["subject"], inplace=True)

    elif site == "1000brains":
        x["participant_id"] = x["subject"].str.split("_ses").str[0]
        x["participant_id"] = "sub-" + x["participant_id"]

        # Extract the session (ses-X part)
        x["session"] = x["subject"].str.extract(r"(ses-\d+)")
        # keep only the first session
        x.query("session == 'ses-1'", inplace=True)
        x.drop(columns=["session", "subject"], inplace=True)

    elif site == "AOMIC_ID1000":
        x["participant_id"] = x["subject"].str.split("_run").str[0]
        x["participant_id"] = "sub-" + x["participant_id"]
        x["session"] = x["subject"].str.extract(r"(run-\d+)")
        x.query("session == 'run-1'", inplace=True)
        x.drop(columns=["session", "subject"], inplace=True)

    elif site == "GSP":
        x["participant_id"] = x["subject"].str.split("_ses").str[0]
        x["participant_id"] = "sub-" + x["participant_id"]

        # Extract the session (ses-X part)
        x["session"] = x["subject"].str.extract(r"(ses-\d+)")
        # keep only the first session
        x.query("session == 'ses-1'", inplace=True)
        x.drop(columns=["session", "subject"], inplace=True)

    elif site == "DLBS":
        x["participant_id"] = x["subject"].str.split("_ses").str[0]
        x["participant_id"] = "sub-" + x["participant_id"]

        # Extract the session (ses-X part)
        x["session"] = x["subject"].str.extract(r"(ses-\d+)")
        # keep only the first session
        x.query("session == 'ses-1'", inplace=True)
        x.drop(columns=["session", "subject"], inplace=True)

    return x


def processing_participant_tsv(participants, site):
    participants.columns = participants.columns.str.lower()

    if site == "DLBS":
        participants["participant+af8-id"] = participants[
            "participant+af8-id"
        ].str.lower()
        participants["participant_id"] = (
            "sub-"
            + participants["participant+af8-id"]
            .str.split("sub+")
            .str[1]
            .str.split("-")
            .str[1]
        )

        participants = participants.loc[:, ["participant_id", "age", "gender"]]

    elif site == "GSP":
        participants["subject_id"] = participants["subject_id"].str.lower()

        participants["participant_id"] = (
            participants["subject_id"].str.split("_ses").str[0]
        )
        participants["participant_id"] = (
            "sub-" + participants["participant_id"].str.split("sub").str[1]
        )

        participants["participant_id"] = participants["participant_id"]
        # Extract the session (ses-X part)
        participants["session"] = participants["subject_id"].str.extract(r"(ses\d+)")
        participants.query("session == 'ses1'", inplace=True)

        participants = participants.loc[:, ["participant_id", "age_bin", "sex"]]
        participants.rename(columns={"sex": "gender"}, inplace=True)
        participants.rename(columns={"age_bin": "age"}, inplace=True)

    elif site == "AOMIC_ID1000":
        participants = participants.loc[:, ["participant_id", "sex", "age"]]
        participants.rename(columns={"sex": "gender"}, inplace=True)

    elif site == "1000brains":
        participants = participants.loc[:, ["participant_id", "age", "sex"]]
        participants.rename(columns={"sex": "gender"}, inplace=True)

    elif site == "eNKI":
        participants.query("session == 'ses-clga'", inplace=True)
        participants = participants.loc[:, ["participant_id", "age", "sex"]]
        participants.rename(columns={"sex": "gender"}, inplace=True)
    elif site == "SALD":
        participants["participant_id"] = (
            "sub-0" + participants["participant_id"].str.split("sub-").str[1]
        )
        participants = participants.loc[:, ["participant_id", "age", "sex"]]
        participants.rename(columns={"sex": "gender"}, inplace=True)

    return participants

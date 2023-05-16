"""
Determine study groups:

- Medical Intensive Care Unit (MICU)
- Medical/Surgical Intensive Care Unit (MICU/SICU)
- Cardiac Vascular Intensive Care Unit (CVICU)
- Surgical Intensive Care Unit (SICU)
- Trauma SICU (TSICU)
- Coronary Care Unit (CCU)
- Neuro Intermediate
- Neuro Surgical Intensive Care Unit (Neuro SICU)
- Neuro Stepdown

- ECMO
"""

import pandas as pd

if __name__ == "__main__":
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")

    # Get df by careunit
    icustays = pd.get_dummies(
        icustays[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "intime",
                "outtime",
                "los",
                "first_careunit",
            ]
        ],
        columns=["first_careunit"],
        prefix="",
        prefix_sep="",
    )

    # Get ECMO label
    ecmo_events = pd.read_parquet("cache/ecmoevents.parquet")
    ecmo_stayids = ecmo_events["stay_id"].unique()

    icustays["ECMO"] = icustays["stay_id"].apply(lambda sid: int(sid in ecmo_stayids))

    # drop double stays if not ECMO
    has_multiple_stays = icustays.groupby("hadm_id").apply(lambda g: len(g) > 1)
    multiple_stay_hadms = has_multiple_stays[has_multiple_stays].index.to_list()
    print(
        f"[*] Found {len(multiple_stay_hadms)} hospital admissions with multiple ICU stays"
    )

    icustays = icustays[
        (~icustays["hadm_id"].isin(multiple_stay_hadms)) | (icustays["ECMO"] == 1)
    ]

    icustays.to_parquet("cache/studygroups.parquet", index=False)

    print("done")

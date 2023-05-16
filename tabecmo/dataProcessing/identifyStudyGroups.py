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


def get_hadm_label_map(hadm_ids: list):
    diagnoses = pd.read_csv("mimiciv/hosp/diagnoses_icd.csv")
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)]

    # Thrombosis
    diagnoses["comp_dvt"] = (diagnoses["icd_code"].str.startswith("I8240")) | (
        diagnoses["icd_code"].str.startswith("4534")
    )

    diagnoses["comp_pe"] = (diagnoses["icd_code"].str.startswith("I26")) | (
        diagnoses["icd_code"].str.startswith("4151")
    )

    diagnoses["comp_ox_thrombosis"] = diagnoses["icd_code"].isin(["T82867A", "99672"])

    diagnoses["comp_retinal_vascular_occlusion"] = (
        diagnoses["icd_code"].str.startswith("3623")
    ) | (diagnoses["icd_code"].str.startswith("H341"))

    diagnoses["comp_precerebral_occlusion"] = (
        diagnoses["icd_code"].str.startswith("433")
    ) | (diagnoses["icd_code"].str.startswith("I63"))

    diagnoses["comp_cerebral_occlusion"] = (
        diagnoses["icd_code"].str.startswith("434")
    ) | (diagnoses["icd_code"].str.startswith("I64"))

    diagnoses["comp_unspec_thrombosis"] = (diagnoses["icd_code"] == "4449") | (
        diagnoses["icd_code"].str.startswith("I74")
    )

    diagnoses["comp_any_thrombosis"] = (
        diagnoses["comp_dvt"]
        | diagnoses["comp_pe"]
        | diagnoses["comp_ox_thrombosis"]
        | diagnoses["comp_retinal_vascular_occlusion"]
        | diagnoses["comp_precerebral_occlusion"]
        | diagnoses["comp_cerebral_occlusion"]
        | diagnoses["comp_unspec_thrombosis"]
    )

    # Hemorrhage
    diagnoses["comp_ic_hemorrhage"] = (diagnoses["icd_code"].str.startswith("431")) | (
        diagnoses["icd_code"].str.startswith("I61")
    )

    diagnoses["comp_gi_hemorrhage"] = diagnoses["icd_code"].isin(
        ["578", "5780", "5781", "5789", "K922"]
    )

    diagnoses["comp_csite_hemorrhage"] = diagnoses["icd_code"].isin(
        ["99674", "T82838A"]
    )
    diagnoses["comp_ssite_hemorrhage"] = diagnoses["icd_code"].isin(["L7622", "99811"])
    diagnoses["comp_pulm_hemorrhage"] = diagnoses["icd_code"].isin(["R0489", "78639"])
    diagnoses["comp_epistaxis"] = diagnoses["icd_code"].isin(["R040", "7847"])
    diagnoses["comp_unspec_hemorrhage"] = diagnoses["icd_code"].isin(
        ["99811", "I97418", "I9742", "I97618", "I9762"]
    )

    diagnoses["comp_subarachnoid_hemorrhage"] = (
        diagnoses["icd_code"].str.startswith("430")
    ) | (diagnoses["icd_code"].str.startswith("I60"))

    diagnoses["comp_any_hemorrhage"] = (
        diagnoses["comp_ic_hemorrhage"]
        | diagnoses["comp_subarachnoid_hemorrhage"]
        | diagnoses["comp_gi_hemorrhage"]
        | diagnoses["comp_csite_hemorrhage"]
        | diagnoses["comp_ssite_hemorrhage"]
        | diagnoses["comp_pulm_hemorrhage"]
        | diagnoses["comp_epistaxis"]
        | diagnoses["comp_unspec_hemorrhage"]
    )

    # Stroke
    # ahajournals.org/doi/10.1161/01.str.0000174293.17959.a1
    diagnoses["comp_other_acute_stroke"] = diagnoses["icd_code"].str.startswith("436")
    diagnoses["comp_tia"] = (diagnoses["icd_code"].str.startswith("435")) | (
        diagnoses["icd_code"].str.startswith("G45")
    )

    diagnoses["comp_any_stroke"] = (
        diagnoses["comp_other_acute_stroke"]
        | diagnoses["comp_tia"]
        | diagnoses["comp_retinal_vascular_occlusion"]
        | diagnoses["comp_precerebral_occlusion"]
        | diagnoses["comp_cerebral_occlusion"]
        | diagnoses["comp_ic_hemorrhage"]
        | diagnoses["comp_subarachnoid_hemorrhage"]
    )

    complication_cols = [c for c in diagnoses.columns if c.startswith("comp_")]
    return diagnoses[complication_cols + ["hadm_id"]].groupby("hadm_id").agg("max")


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

    # drop stays < 12 hrs if not ECMO
    icustays = icustays[(icustays["los"] > 0.5) | (icustays["ECMO"] == 1)]

    # Labels
    print("[*] Getting labels...")
    hadm_label_map = get_hadm_label_map(icustays["hadm_id"].to_list())

    # NOTE: implicitly dropping anything w/no listed diagnoses
    icustays = pd.merge(
        icustays, hadm_label_map, how="right", left_on="hadm_id", right_index=True
    )

    icustays.to_parquet("cache/studygroups.parquet", index=False)

    print("done")

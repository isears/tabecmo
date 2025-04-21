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


def apply_filter(fn: callable, df_in) -> pd.DataFrame:
    before_count = len(df_in)
    df_out = fn(df_in)
    after_count = len(df_out)
    print(
        f"Applied {fn.__name__}: {before_count} -> {after_count} (dropped {before_count - after_count})"
    )

    return df_out


if __name__ == "__main__":
    icustays = pd.read_csv("mimiciv/icu/icustays.csv")
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

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
        prefix="unit",
    )

    # Get ECMO label
    ecmo_events = pd.read_parquet("cache/ecmoevents.parquet")
    ecmo_stayids = ecmo_events["stay_id"].unique()
    first_cannulation_events = (
        ecmo_events[(ecmo_events["itemid"].isin([229268, 229840]))]
        .groupby("stay_id")["charttime"]
        .agg("min")
    )

    ecmo_stays = icustays[icustays["stay_id"].isin(ecmo_stayids)]
    print(f"[*] Starting with {len(ecmo_stays)} ecmo stays")

    ecmo_stays = pd.merge(
        ecmo_stays,
        first_cannulation_events.rename("cannulationtime"),
        how="left",
        left_on="stay_id",
        right_index=True,
    )

    def drop_short_stays(df_in: pd.DataFrame):
        return df_in[df_in["los"] > 2]

    ecmo_stays = apply_filter(drop_short_stays, ecmo_stays)

    def drop_subsequent_stays_for_subject(df_in: pd.DataFrame):
        # If a subject has multiple ICU stays, drop all but first
        return df_in.groupby("subject_id", group_keys=False).apply(
            lambda g: g[g["intime"] == g["intime"].max()]
        )

    ecmo_stays = apply_filter(drop_subsequent_stays_for_subject, ecmo_stays)

    print(f"[*] Starting with {len(icustays)} general ICU stays")

    # In general ICU population, drop ECMO stays
    def drop_ecmo_subjects(df_in: pd.DataFrame):
        return df_in[~df_in["subject_id"].isin(ecmo_stays["subject_id"])]

    # Some icustay times are NaT
    def drop_nat_icustays(df_in: pd.DataFrame):
        return df_in.dropna(subset=["intime", "outtime"], how="any")

    general_stays = apply_filter(drop_short_stays, icustays)
    general_stays = apply_filter(drop_nat_icustays, general_stays)
    general_stays = apply_filter(drop_ecmo_subjects, general_stays)
    general_stays = apply_filter(drop_subsequent_stays_for_subject, general_stays)

    general_stays["ECMO"] = 0
    ecmo_stays["ECMO"] = 1

    final_study_group = pd.concat([general_stays, ecmo_stays])

    final_study_group.to_parquet("cache/studygroups.parquet", index=False)

    print("done")

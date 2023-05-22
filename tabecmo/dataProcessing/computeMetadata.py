import pandas as pd

from tabecmo.dataProcessing import features
from tabecmo.dataProcessing.aggregateDerivedTables import (
    meds_tables,
    standard_tables,
    zero_info_columns,
)

if __name__ == "__main__":
    stats = list()

    # Standard tables
    for table_name in standard_tables + ["bg", "vitalsign"]:
        df = pd.read_parquet(f"mimiciv_derived/{table_name}.parquet")

        if "temperature" in df.columns:
            df["temperature"] = df["temperature"].astype("float")

        aggable_columns = [
            c
            for c in df.columns
            if c
            not in [
                "charttime",
                "subject_id",
                "stay_id",
                "hadm_id",
                "specimen_id",
                "specimen",
                "temperature_site",
            ]
            + zero_info_columns
        ]

        stats.append(df[aggable_columns].agg(["mean", "median", "std", "max", "min"]))

    all_stats = pd.concat(stats, axis="columns")

    # One-offs
    all_stats["fio2"] = all_stats[["fio2", "fio2_chartevents"]].mean(axis=1)
    all_stats["systolic_bp"] = all_stats[["sbp", "sbp_ni"]].mean(axis=1)
    all_stats["diastolic_bp"] = all_stats[["dbp", "dbp_ni"]].mean(axis=1)

    all_stats = all_stats.groupby(by=all_stats.columns, axis=1).mean()
    # TODO: some columns have no values ever; should stop using these entirely
    all_stats = all_stats.fillna(0.0)

    all_stats.to_parquet("mimiciv_derived/processed/stats.parquet")
    # Drop unused columnns; reorder used columns
    all_stats = all_stats[[f for f in features if f not in meds_tables]]

    print("done")

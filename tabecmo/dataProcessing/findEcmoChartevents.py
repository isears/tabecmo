"""
Build ecmoevents.parquet
"""

import datetime

import dask.dataframe as dd
import pandas as pd

if __name__ == "__main__":
    chartevents_dd = dd.read_csv(
        "mimiciv/icu/chartevents.csv",
        usecols=["itemid", "stay_id", "valuenum", "value", "charttime"],
        dtype={
            "itemid": "int",
            "stay_id": "int",
            "valuenum": "float",
            "value": "object",
            "charttime": "object",
        },
    )

    ecmo_circuit_config_ids = [229268, 229840]
    ecmo_flow_ids = [229270, 229842]

    # Most restrictive filtering: must have a configuration event and at least one flow measurement
    ecmo_events = chartevents_dd[
        chartevents_dd["itemid"].isin(ecmo_flow_ids + ecmo_circuit_config_ids)
    ].compute(scheduler="processes")

    ecmo_events["charttime"] = pd.to_datetime(ecmo_events["charttime"])

    ecmo_events.to_parquet("cache/ecmoevents.parquet", index=False)

    print("done")

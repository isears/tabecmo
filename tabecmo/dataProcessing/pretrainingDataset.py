import numpy as np
import pandas as pd
import torch

from tabecmo.dataProcessing.derivedDataset import DerivedDataset


class ImputationDataset(DerivedDataset):
    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]
        X = self.__getitem_X__(stay_id)

        # TODO: this should be coupled with inclusion criteria
        assert (
            len(X.index) >= 48
        ), f"{stay_id} doesn't have sufficient data for imputation"

        x_past = X.iloc[:-24]
        x_future = X.iloc[-24:]

        # average values for the next 24 hrs to make target as rich as possible
        x_future_meds = x_future[self.meds_tables].max()
        non_meds_columns = [c for c in X.columns if c not in self.meds_tables]

        x_future_measurements = x_future[non_meds_columns].replace(-1, np.nan)
        x_future_measurements = x_future_measurements.mean().fillna(-1)

        x_future_summarized = pd.concat([x_future_measurements, x_future_meds])

        return torch.tensor(x_past.transpose().to_numpy()), torch.tensor(
            x_future_summarized.to_numpy()
        )


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["los"] > 2)
    ]
    sids = studygroups["stay_id"].to_list()
    ds = ImputationDataset(sids)

    print(ds[0])

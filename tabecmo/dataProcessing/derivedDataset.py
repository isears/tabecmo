import os
import random

import pandas as pd
import torch
from torch.nn.functional import pad

from tabecmo import config
from tabecmo.dataProcessing import features
from tabecmo.dataProcessing.aggregateDerivedTables import meds_tables


class DerivedDataset(torch.utils.data.Dataset):
    used_tables = [
        "vitalsign",
        "chemistry",
        "coagulation",
        "blood_differential",
        "bg",
        "enzyme",
        "inflammation",
        "dobutamine",
        "epinephrine",
        "invasive_line",
        "milrinone",
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
        "ventilation",
    ]

    def __init__(
        self,
        stay_ids: list[int],
        shuffle: bool = True,
        pm_type=torch.bool,  # May require pad mask to be different type
    ):
        print(f"[{type(self).__name__}] Initializing dataset...")
        self.stay_ids = stay_ids

        if shuffle:
            random.shuffle(self.stay_ids)

        print(f"\tExamples: {len(self.stay_ids)}")

        self.features = features
        print(f"\tFeatures: {len(self.features)}")

        self.pm_type = pm_type
        # TODO:
        # self.max_len = self.examples["cutidx"].max() + 1
        # print(f"\tMax length: {self.max_len}")
        self.max_len = 50 * 24

        self.stats = pd.read_parquet("mimiciv_derived/processed/stats.parquet")

        self.labels = pd.read_parquet("mimiciv_derived/icustay_detail.parquet")
        self.labels = self.labels.set_index("stay_id")["hospital_expire_flag"].astype(
            int
        )

    def maxlen_padmask_collate(self, batch):
        """
        Pad and return third value (the pad mask)
        Returns X, y, padmask, stay_ids
        """
        for idx, (X, y) in enumerate(batch):
            # X = torch.tensor(X.transpose().to_numpy())
            actual_len = X.shape[1]

            assert (
                actual_len <= self.max_len
            ), f"Actual: {actual_len}, Max: {self.max_len}"

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, y, pad_mask)

        X = torch.stack([X for X, _, _ in batch], dim=0)
        y = torch.stack([Y for _, Y, _ in batch], dim=0)  # .unsqueeze(-1)

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        pad_mask = torch.stack([pad_mask for _, _, pad_mask in batch], dim=0)

        return X.float(), y.float(), pad_mask.to(self.pm_type)

    def __getitem_X__(self, stay_id: int) -> pd.DataFrame:
        loaded_dfs = list()

        for table_name in self.used_tables:
            if os.path.exists(
                f"mimiciv_derived/processed/{stay_id}.{table_name}.parquet"
            ):
                loaded_dfs.append(
                    pd.read_parquet(
                        f"mimiciv_derived/processed/{stay_id}.{table_name}.parquet"
                    )
                )

        combined = pd.concat(loaded_dfs, axis="columns")
        # Different data sources may measure same values (for example, blood gas and chemistries)
        # When that happens, just take the mean
        combined = combined.T.groupby(by=combined.columns).mean().transpose()

        # Min / max normalization
        for col in combined.columns:
            if col in self.stats.columns:
                combined[col] = (combined[col] - self.stats[col].loc["min"]) / (
                    self.stats[col].loc["max"] - self.stats[col].loc["min"]
                )

        combined = combined.reindex(columns=self.features)

        # Fill meds related missing values w/0.0, b/c missingness implies drugs were not administered
        combined[meds_tables] = combined[meds_tables].fillna(0.0)
        # Fill all other missing values w/-1.0, b/c these values are truly missing
        combined = combined.fillna(-1.0)

        return combined

    def __getitem_Y__(self, stay_id: int) -> float:
        return self.labels.loc[stay_id].astype(int)

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)
        y = self.__getitem_Y__(stay_id)

        return torch.tensor(X.transpose().to_numpy()), torch.tensor(y.to_numpy())


class SnapshotDataset(DerivedDataset):
    """
    Instead of returning the entire time series, return a snapshot of the most recent available
    information 24 hrs prior to the end of the ICU stay
    """

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X_complete = self.__getitem_X__(stay_id)
        # Truncates ICU stays to 24 hrs before end
        X = X_complete.iloc[:-24]
        X_ffill = (
            X.map(lambda item: float("nan") if item == -1 else item).ffill().fillna(-1)
        )

        y = self.__getitem_Y__(stay_id)

        return torch.tensor(X_ffill.transpose().to_numpy()[:, -1]), torch.tensor(y)


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["los"] > 2)
    ]
    sids = studygroups["stay_id"].to_list()

    ds = SnapshotDataset(sids)
    print(ds[0])

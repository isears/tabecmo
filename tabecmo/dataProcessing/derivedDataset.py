import datetime
import glob
import os
import random

import pandas as pd
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from tabecmo import config

random.seed(42)


class DerivedDataset(torch.utils.data.Dataset):
    used_tables = [
        "vitalsign",
        "chemistry",
        "coagulation",
        "differential_detailed",
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

    meds_tables = [
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

        self.features = self._populate_features()
        print(f"\tFeatures: {len(self.features)}")

        self.pm_type = pm_type
        # TODO:
        # self.max_len = self.examples["cutidx"].max() + 1
        # print(f"\tMax length: {self.max_len}")
        self.max_len = 14 * 24 + 1

        self.stats = pd.read_parquet("mimiciv_derived/processed/stats.parquet")

    def _populate_features(self) -> list:
        feature_names = list()

        for table_name in self.used_tables:
            example_table = pd.read_parquet(
                glob.glob(f"mimiciv_derived/processed/*.{table_name}.parquet")[0]
            )

            feature_names += example_table.columns.to_list()

        return sorted(list(set(feature_names)))

    def maxlen_padmask_collate(self, batch):
        """
        Pad and return third value (the pad mask)
        Returns X, y, padmask, stay_ids
        """
        for idx, (X, y, stay_id) in enumerate(batch):
            X = torch.tensor(X.transpose().to_numpy())
            actual_len = X.shape[1]

            assert (
                actual_len <= self.max_len
            ), f"Actual: {actual_len}, Max: {self.max_len}"

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, y, pad_mask, stay_id)

        X = torch.stack([X for X, _, _, _ in batch], dim=0)
        y = torch.stack(
            [torch.tensor(Y) for _, Y, _, _ in batch], dim=0
        )  # .unsqueeze(-1)
        pad_mask = torch.stack([pad_mask for _, _, pad_mask, _ in batch], dim=0)
        IDs = torch.tensor([stay_id for _, _, _, stay_id in batch])

        return X.float(), y.float(), pad_mask.to(self.pm_type), IDs

    def maxlen_padmask_collate_skorch(self, batch):
        """
        Skorch expects kwargs output
        """
        X, y, pad_mask, _ = self.maxlen_padmask_collate(batch)
        return dict(X=X, padding_masks=pad_mask), y

    def last_available_collate(self, batch):
        for idx, (X, y, stay_id) in enumerate(batch):
            non_med_cols = [c for c in X.columns if c not in self.meds_tables]
            med_cols = [c for c in X.columns if c in self.meds_tables]

            mask = pd.concat(
                [(X[non_med_cols] == -1), ((X[med_cols] == 0) | (X[med_cols] == -1))],
                axis="columns",
            )

            last_available_x = X.mask(mask).ffill().iloc[[-1]]
            # TODO: how should we fill missing?
            last_available_x[non_med_cols] = last_available_x[non_med_cols].fillna(0.0)
            last_available_x[med_cols] = last_available_x[med_cols].fillna(0.0)

            last_available_x = torch.tensor(last_available_x.to_numpy())
            batch[idx] = (last_available_x, y, stay_id)

        X = torch.stack([X for X, _, _ in batch], dim=0)
        X = torch.squeeze(X, dim=1)
        y = torch.stack([torch.tensor(Y) for _, Y, _ in batch], dim=0)  # .unsqueeze(-1)

        return X.float(), y.float()

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
        combined = combined.groupby(by=combined.columns, axis=1).mean()

        # Min / max normalization
        for col in combined.columns:
            if col in self.stats.columns:
                combined[col] = (combined[col] - self.stats[col].loc["min"]) / (
                    self.stats[col].loc["max"] - self.stats[col].loc["min"]
                )

        combined = combined.reindex(columns=self.features)

        # Fill nas w/-1
        # TODO: if no drug data exists, need to fill those nas w/0.0
        combined = combined.fillna(-1.0)

        return combined

    def __getitem_Y__(self, stay_id: int) -> float:
        if os.path.exists(f"mimiciv_derived/processed/{stay_id}.sepsis3.parquet"):
            return 1.0
        else:
            return 0.0

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)
        Y = self.__getitem_Y__(stay_id)

        if Y == 1.0:  # Do just-before-sepsis cut
            sepsis_df = pd.read_parquet(
                f"mimiciv_derived/processed/{stay_id}.sepsis3.parquet"
            )
            t_sepsis = sepsis_df[sepsis_df["sepsis3"] == 1].index[0]
            t_cut = t_sepsis - datetime.timedelta(hours=12)

            if t_cut > X.index[0]:
                daterange = pd.date_range(X.index[0], t_cut, freq="H")
                X = X.loc[daterange]
            else:
                print(
                    f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do pre-sepsis cut"
                )

        else:  # Do random cut
            # TODO: this should be coupled with inclusion criteria
            if len(X.index) <= 24:
                print(
                    f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do random cut"
                )
            else:
                t_cut = random.choice(X.index[24:])
                daterange = pd.date_range(X.index[0], t_cut, freq="H")
                X = X.loc[daterange]

        return X, Y, stay_id


class UnlabeledDataset(DerivedDataset):
    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)

        if len(X.index) < 12:
            print(
                f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do random cut"
            )
        else:
            t_cut = random.choice(X.index[12:])
            daterange = pd.date_range(X.index[0], t_cut, freq="H")
            X = X.loc[daterange]

        X_ffill = (
            X.applymap(lambda item: float("nan") if item == -1 else item)
            .fillna(method="ffill")
            .fillna(-1)
        )

        return torch.tensor(X_ffill.transpose().to_numpy()[:, -1])


def load_to_mem_unsupervised(stay_ids: list):
    all_X = torch.tensor([])

    ds = UnlabeledDataset(stay_ids=stay_ids)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
        pin_memory=True,
    )

    for batch_X in dl:
        all_X = torch.cat((all_X, batch_X))

    return all_X


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["los"] > 0.6)
    ]
    sids = studygroups["stay_id"].to_list()

    X = load_to_mem_unsupervised(sids)

    print(X.shape)

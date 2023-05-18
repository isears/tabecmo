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

    label_names = [
        "comp_dvt",
        "comp_pe",
        "comp_ox_thrombosis",
        "comp_retinal_vascular_occlusion",
        "comp_precerebral_occlusion",
        "comp_cerebral_occlusion",
        "comp_unspec_thrombosis",
        "comp_ic_hemorrhage",
        "comp_gi_hemorrhage",
        "comp_csite_hemorrhage",
        "comp_ssite_hemorrhage",
        "comp_pulm_hemorrhage",
        "comp_epistaxis",
        "comp_unspec_hemorrhage",
        "comp_subarachnoid_hemorrhage",
        "comp_other_acute_stroke",
        "comp_tia",
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
        self.max_len = 50 * 24

        self.stats = pd.read_parquet("mimiciv_derived/processed/stats.parquet")

        self.labels = pd.read_parquet("cache/studygroups.parquet")
        self.labels = self.labels.set_index("stay_id")
        self.labels = self.labels[
            [c for c in self.labels.columns if c.startswith("comp_")]
        ]

        # For now, use the broad categories
        self.labels = self.labels.reindex(columns=self.label_names)

    def _populate_features(self) -> list:
        # TODO: hardcode for now but need to think of a way to generate this more efficiently
        return [
            "AbsoluteBasophilCount",
            "AbsoluteEosinophilCount",
            "AbsoluteLymphocyteCount",
            "AbsoluteMonocyteCount",
            "AbsoluteNeutrophilCount",
            "AtypicalLymphocytes",
            "Bands",
            "Basophils",
            "Blasts",
            "EosinophilCount",
            "Eosinophils",
            "GranulocyteCount",
            "HypersegmentedNeutrophils",
            "ImmatureGranulocytes",
            "Lymphocytes",
            "LymphocytesPercent",
            "Metamyelocytes",
            "MonocyteCount",
            "Monocytes",
            "Myelocytes",
            "Neutrophils",
            "NucleatedRedCells",
            "OtherCells",
            "PlateletCount",
            "Promyelocytes",
            "RedBloodCells",
            "ReticulocyteCountAbsolute",
            "ReticulocyteCountAutomated",
            "WBCCount",
            "WhiteBloodCells",
            "aado2",
            "aado2_calc",
            "albumin",
            "alp",
            "alt",
            "amylase",
            "aniongap",
            "ast",
            "baseexcess",
            "bicarbonate",
            "bilirubin_direct",
            "bilirubin_indirect",
            "bilirubin_total",
            "bun",
            "calcium",
            "carboxyhemoglobin",
            "chloride",
            "ck_cpk",
            "ck_mb",
            "creatinine",
            "crp",
            "d_dimer",
            "diastolic_bp",
            "dobutamine",
            "epinephrine",
            "fibrinogen",
            "fio2",
            "ggt",
            "globulin",
            "glucose",
            "heart_rate",
            "hematocrit",
            "hemoglobin",
            "inr",
            "invasive_line",
            "lactate",
            "ld_ldh",
            "methemoglobin",
            "milrinone",
            "norepinephrine",
            "pao2fio2ratio",
            "pco2",
            "ph",
            "phenylephrine",
            "po2",
            "potassium",
            "pt",
            "ptt",
            "resp_rate",
            "so2",
            "sodium",
            "spo2",
            "systolic_bp",
            "temperature",
            "thrombin",
            "total_protein",
            "totalco2",
            "vasopressin",
            "ventilation",
        ]
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
        combined[self.meds_tables] = combined[self.meds_tables].fillna(0.0)
        combined = combined.fillna(-1.0)

        return combined

    def __getitem_Y__(self, stay_id: int) -> float:
        return self.labels.loc[stay_id].astype(int)

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)
        # Y = self.__getitem_Y__(stay_id)

        # TODO: this should be coupled with inclusion criteria
        if len(X.index) <= 24:
            print(
                f"[-] Warning: stay id {stay_id} doesn't have sufficient data to do random cut"
            )
        else:
            t_cut = random.choice(X.index[24:])
            daterange = pd.date_range(X.index[0], t_cut, freq="H")
            X = X.loc[daterange]

        y = self.__getitem_Y__(stay_id)

        return torch.tensor(X.transpose().to_numpy()), torch.tensor(y.to_numpy())


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


class UnlabeledTimeseriesDataset(DerivedDataset):
    """
    Do random cut, but deliver entire timeseries up to cut rather than just most recent data
    """

    def maxlen_padmask_collate(self, batch):
        """
        Pad and return third value (the pad mask)
        Returns X, y, padmask, stay_ids
        """
        for idx, X in enumerate(batch):
            actual_len = X.shape[1]

            assert (
                actual_len <= self.max_len
            ), f"Actual: {actual_len}, Max: {self.max_len}"

            pad_mask = torch.ones(actual_len)
            X_mod = pad(X, (0, self.max_len - actual_len), mode="constant", value=0.0)

            pad_mask = pad(
                pad_mask, (0, self.max_len - actual_len), mode="constant", value=0.0
            )

            batch[idx] = (X_mod.T, pad_mask)

        X = torch.stack([x for x, _ in batch], dim=0)
        pad_mask = torch.stack([pad_mask for _, pad_mask in batch], dim=0)

        return X.float(), pad_mask.to(self.pm_type)

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

        return torch.tensor(X.transpose().to_numpy())


class LabeledEcmoDataset(DerivedDataset):
    def __init__(self):
        self.ecmoevents = pd.read_parquet("cache/ecmoevents.parquet")

        studygroup = pd.read_parquet("cache/studygroups.parquet")
        ecmo_stayids = studygroup[studygroup["ECMO"] == 1]["stay_id"].to_list()

        super().__init__(ecmo_stayids)

    def __getitem_X__(self, stay_id: int):
        X = super().__getitem_X__(stay_id)
        cannulation_events = self.ecmoevents[
            (self.ecmoevents["stay_id"] == stay_id)
            & (self.ecmoevents["itemid"].isin([229268, 229840]))
        ]

        cannulation_time = cannulation_events["charttime"].min()
        cannulation_idx = (X.index > cannulation_time).tolist().index(True)
        X = X.iloc[:cannulation_idx]
        return X

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)

        y = torch.tensor(self.labels.loc[stay_id].astype(int).to_numpy())
        return torch.tensor(X.transpose().to_numpy()), y


class LabeledEcmoDatasetTruncated(LabeledEcmoDataset):
    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X = self.__getitem_X__(stay_id)

        X_ffill = (
            X.applymap(lambda item: float("nan") if item == -1 else item)
            .fillna(method="ffill")
            .fillna(-1)
        )

        y = torch.tensor(self.labels.loc[stay_id].astype(int).to_numpy())

        return torch.tensor(X_ffill.transpose().to_numpy()[:, -1]), y


class IhmLabelingDataset(DerivedDataset):
    def __init__(self, stay_ids: list[int], shuffle: bool = True, pm_type=torch.bool):
        super().__init__(stay_ids, shuffle, pm_type)

        self.labels = pd.read_parquet("mimiciv_derived/icustay_detail.parquet")
        self.labels = self.labels.set_index("stay_id")["hospital_expire_flag"].astype(
            int
        )

    def __getitem_Y__(self, stay_id: int):
        return self.labels.loc[stay_id]

    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X_complete = self.__getitem_X__(stay_id)
        assert len(X_complete) > 48

        X = X_complete.iloc[:-24]
        y = self.__getitem_Y__(stay_id)

        return torch.tensor(X.transpose().to_numpy()), torch.tensor(y)


class IhmLabelingDatasetTruncated(IhmLabelingDataset):
    def __getitem__(self, index: int):
        stay_id = self.stay_ids[index]

        X_complete = self.__getitem_X__(stay_id)
        X = X_complete.iloc[:-24]
        X_ffill = (
            X.applymap(lambda item: float("nan") if item == -1 else item)
            .fillna(method="ffill")
            .fillna(-1)
        )

        y = self.__getitem_Y__(stay_id)

        return torch.tensor(X_ffill.transpose().to_numpy()[:, -1]), torch.tensor(y)


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
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["los"] > 2)
    ]
    sids = studygroups["stay_id"].to_list()

    ds = IhmLabelingDataset(sids)
    print(ds[0])

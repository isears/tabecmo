from typing import List

import pandas as pd


class Table1Generator(object):
    def __init__(self, stay_ids: List[int]) -> None:
        self.stay_ids = stay_ids
        self.table1 = pd.DataFrame(columns=["Item", "Value"])

        self.all_df = pd.read_parquet("cache/studygroups.parquet")
        self.all_df = self.all_df[self.all_df["stay_id"].isin(stay_ids)]
        self.total_stays = len(self.all_df.index)

        # Create df with all demographic data
        self.all_df = self.all_df.merge(
            pd.read_parquet(
                "mimiciv_derived/icustay_detail.parquet",
                columns=[
                    "stay_id",
                    "gender",
                    "admission_age",
                    "race",
                    "hospital_expire_flag",
                ],
            ),
            how="left",
            on=["stay_id"],
        )

        charlson = pd.read_parquet("mimiciv_derived/charlson.parquet")
        charlson = charlson.drop(
            columns=[c for c in charlson.columns if c in ["subject_id", "age_score"]]
        )

        # Save the available charlson comorbidity categories for use later
        self.charlson_categories = [
            c
            for c in charlson.columns
            if c not in ["hadm_id", "charlson_comorbidity_index"]
        ]

        self.all_df = pd.merge(
            self.all_df,
            charlson,
            how="left",
            on="hadm_id",
        )

        # Reverse the one-hot encoding of the ICU units
        unit_cols = [c for c in self.all_df.columns if c.startswith("unit_")]
        self.all_df["unit"] = (
            self.all_df[unit_cols].idxmax(axis=1).apply(lambda x: x[5:])
        )
        self.all_df = self.all_df.drop(columns=unit_cols)

        # Make sure there's only one stay id per entry so we can confidently calculate statistics
        assert len(self.all_df["stay_id"]) == self.all_df["stay_id"].nunique()

    def _add_table_row(self, item: str, value: str):
        self.table1.loc[len(self.table1.index)] = [item, value]

    def _pprint_percent(self, n: int, total: int = None) -> str:
        if total == None:
            total = self.total_stays

        return f"{n}, ({n / total * 100:.2f} %)"

    def _pprint_mean(self, values: pd.Series):
        return f"{values.mean():.2f} (median {values.median():.2f}, std {values.std():.2f})"

    def _tablegen_count(self) -> None:
        self._add_table_row(
            item="Total Stays", value=self._pprint_percent(len(self.all_df))
        )

    def _tablegen_general_demographics(self) -> None:
        for demographic_name in [
            "gender",
            "race",
            "hospital_expire_flag",
        ]:
            for key, value in (
                self.all_df[demographic_name].value_counts().to_dict().items()
            ):
                self._add_table_row(
                    f"[{demographic_name}] {key}", self._pprint_percent(value)
                )

    def _tablegen_age(self) -> None:
        self._add_table_row(
            item="Average Age at ICU Admission",
            value=self._pprint_mean(self.all_df["admission_age"]),
        )

    def _tablegen_comorbidities(self) -> None:
        for c in self.charlson_categories:
            comorbidity_count = self.all_df[c].sum()

            self._add_table_row(
                f"[comorbidity] {c}", self._pprint_percent(comorbidity_count)
            )

        self._add_table_row(
            item="[comorbidity] Average CCI",
            value=self._pprint_mean(self.all_df["charlson_comorbidity_index"]),
        )

    def _tablegen_LOS(self) -> None:
        self._add_table_row(
            item="Average Length of ICU Stay (days)",
            value=self._pprint_mean(self.all_df["los"]),
        )

    def _tablegen_unit(self) -> None:
        for unit_name, count in self.all_df["unit"].value_counts().to_dict().items():
            self._add_table_row(
                item=f"[unit] {unit_name}", value=self._pprint_percent(count)
            )

    def _tablegen_ECMO(self) -> None:
        ecmo_count = self.all_df["ECMO"].sum()
        self._add_table_row(item="ECMO", value=self._pprint_percent(ecmo_count))

    def populate(self) -> pd.DataFrame:
        tablegen_methods = [m for m in dir(self) if m.startswith("_tablegen")]

        for method_name in tablegen_methods:
            func = getattr(self, method_name)
            print(f"[*] {method_name}")
            func()

        return self.table1


if __name__ == "__main__":
    sg = pd.read_parquet("cache/studygroups.parquet")
    t1generator_all = Table1Generator(sg["stay_id"].to_list())
    t1 = t1generator_all.populate()

    print(t1)

    t1.to_csv("results/table1_all.csv", index=False)

    ecmo_stays = sg[sg["ECMO"] == 1]["stay_id"].to_list()
    t1generator_ECMO = Table1Generator(ecmo_stays)
    t1 = t1generator_ECMO.populate()

    print(t1)

    t1.to_csv("results/table1_ECMO.csv", index=False)

    nonecmo_stays = sg[sg["ECMO"] == 0]["stay_id"].to_list()
    t1generator_ECMO = Table1Generator(nonecmo_stays)
    t1 = t1generator_ECMO.populate()

    print(t1)

    t1.to_csv("results/table1_nonECMO.csv", index=False)

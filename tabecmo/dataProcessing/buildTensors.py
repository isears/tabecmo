"""
Build tensors from datasets
"""

import pandas as pd
import torch

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import LabeledEcmoDataset, UnlabeledDataset

if __name__ == "__main__":
    studygroup = pd.read_parquet("cache/studygroups.parquet")

    # Build unlabeled tensors for each ICU group
    for unit in [c for c in studygroup.columns if c.startswith("unit_")]:
        sids = studygroup[(studygroup[unit] == 1) & (studygroup["ECMO"] == 0)][
            "stay_id"
        ].to_list()

        unit = unit.replace("unit_", "")
        unit = unit.replace(" ", ".")

        if "(" in unit:
            cut_idx = unit.index("(")
            unit = unit[: cut_idx - 1]

        print(f"[*] Loading unlabeled data for {unit}")

        ds = UnlabeledDataset(sids)

        all_X = torch.tensor([])

        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=config.cores_available,
            batch_size=32,
        )

        for batch_X in dl:
            all_X = torch.cat((all_X, batch_X))

        torch.save(all_X, f"cache/X_unlabeled_{unit}.pt")

    # Build X, y tensors for ECMO
    ecmo_ds = LabeledEcmoDataset()

    all_X, all_y = torch.tensor([]), torch.tensor([])

    dl = torch.utils.data.DataLoader(
        ecmo_ds,
        num_workers=config.cores_available,
        batch_size=32,
    )

    for batch_X, batch_y in dl:
        all_X = torch.cat((all_X, batch_X))
        all_y = torch.cat((all_y, batch_y))

    torch.save(all_X, f"cache/X_ecmo.pt")
    torch.save(all_y, f"cache/y_ecmo.pt")

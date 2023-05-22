"""
Build tensors from datasets
"""

import pandas as pd
import torch

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import SnapshotDataset

if __name__ == "__main__":
    studygroup = pd.read_parquet("cache/studygroups.parquet")

    combined_X = torch.tensor([])
    combined_y = torch.tensor([])

    # Build unlabeled tensors for each ICU group
    for unit in [c for c in studygroup.columns if c.startswith("unit_")]:
        stay_ids = studygroup[(studygroup[unit] == 1) & (studygroup["ECMO"] == 0)][
            "stay_id"
        ].to_list()

        unit = unit.replace("unit_", "")
        unit = unit.replace(" ", ".")
        unit = unit.replace("/", "-")

        if "(" in unit:
            cut_idx = unit.index("(")
            unit = unit[: cut_idx - 1]

        print(f"[*] Loading unlabeled data for {unit}")

        ds = SnapshotDataset(stay_ids)

        all_X = torch.tensor([])
        all_y = torch.tensor([])

        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=config.cores_available,
            batch_size=32,
        )

        for batch_X, batch_y in dl:
            all_X = torch.cat((all_X, batch_X))
            all_y = torch.cat((all_y, batch_y))

        torch.save(all_X, f"cache/ihmtensors/X_{unit}.pt")
        torch.save(all_y, f"cache/ihmtensors/y_{unit}.pt")

        combined_X = torch.cat((combined_X, all_X))
        combined_y = torch.cat((combined_y, all_y))

    # Also save a tensor representing all stays combined
    torch.save(combined_X, "cache/ihmtensors/X_combined.pt")
    torch.save(combined_y, "cache/ihmtensors/y_combined.pt")

    # Build X, y tensors for ECMO
    ecmo_stay_ids = studygroup[(studygroup["ECMO"] == 1)]["stay_id"].to_list()
    ecmo_ds = SnapshotDataset(ecmo_stay_ids)

    all_X, all_y = torch.tensor([]), torch.tensor([])

    dl = torch.utils.data.DataLoader(
        ecmo_ds,
        num_workers=config.cores_available,
        batch_size=32,
    )

    for batch_X, batch_y in dl:
        all_X = torch.cat((all_X, batch_X))
        all_y = torch.cat((all_y, batch_y))

    torch.save(all_X, f"cache/ihmtensors/X_ecmo.pt")
    torch.save(all_y, f"cache/ihmtensors/y_ecmo.pt")

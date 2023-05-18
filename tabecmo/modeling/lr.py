import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import IhmLabelingDatasetTruncated


def do_cv(X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        lr = LogisticRegression()
        lr.fit(
            X[train_indices],
            y[train_indices],
        )

        preds = lr.predict_proba(X[test_indices])[:, 1]

        try:
            score = roc_auc_score(y[test_indices], preds)
            scores.append(score)
            print(f"Fold #{fold_idx} {score}")
        except ValueError:  # low-prevalence classes
            print(f"[-] Warning: no positive examples in fold, skipping")

    scores = np.array(scores)
    print(f"[+] Average (std) CV score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    # Load stay ids
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    ecmo_stay_ids = studygroups[
        (studygroups["ECMO"] == 1)
        & (studygroups["los"] > 2)
        & (studygroups["los"] < 50)
    ]["stay_id"].to_list()

    ds = IhmLabelingDatasetTruncated(ecmo_stay_ids)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=len(ds.stay_ids),
        # collate_fn=ds.maxlen_padmask_collate,
        pin_memory=True,
    )

    X, y = next(iter(dl))
    X = X.numpy()
    y = y.numpy()

    n_folds = 5

    do_cv(X, y, n_splits=n_folds)

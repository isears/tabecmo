import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import LabeledEcmoDatasetTruncated


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
    ds = LabeledEcmoDatasetTruncated()
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

    for label_idx, label in enumerate(ds.label_names):
        if y[:, label_idx].sum() < n_folds:  # Need at least one + example / fold
            print(f"[-] Skipping label {label} b/c too few positive examples")
            continue

        print(f"=========== Running CV for target {label}")
        do_cv(X, y[:, label_idx], n_splits=n_folds)

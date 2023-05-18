import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import LabeledEcmoDataset
from tabecmo.modeling.lookaheadTst import GenericPlTst, build_dl
from tabecmo.modeling.lr import do_cv

if __name__ == "__main__":
    ds = LabeledEcmoDataset()
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=len(ds.stay_ids),
        collate_fn=ds.maxlen_padmask_collate,
        pin_memory=True,
    )

    X, y, pad_mask = next(iter(dl))

    with torch.no_grad():
        tst = GenericPlTst.load_from_checkpoint("cache/best_tst").eval()

        X_imputed = tst.forward_partial(X, pad_mask)

    X_imputed = X_imputed.numpy()
    y = y.numpy()

    n_folds = 5

    for label_idx, label in enumerate(ds.label_names):
        if y[:, label_idx].sum() < n_folds:  # Need at least one + example / fold
            print(f"[-] Skipping label {label} b/c too few positive examples")
            continue

        print(f"=========== Running CV for target {label}")
        do_cv(X_imputed, y[:, label_idx], n_splits=n_folds)

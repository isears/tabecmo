import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import IhmLabelingDataset
from tabecmo.modeling.lookaheadTst import GenericPlTst, build_dl
from tabecmo.modeling.lr import do_cv

if __name__ == "__main__":
    # Load stay ids
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    ecmo_stay_ids = studygroups[
        (studygroups["ECMO"] == 1)
        & (studygroups["los"] > 2)
        & (studygroups["los"] < 50)
    ]["stay_id"].to_list()

    ds = IhmLabelingDataset(ecmo_stay_ids)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=len(ds.stay_ids),
        collate_fn=ds.maxlen_padmask_collate,
        pin_memory=True,
    )

    X, y, pad_mask = next(iter(dl))

    with torch.no_grad():
        tst = GenericPlTst.load_from_checkpoint("cache/imputer.ckpt").eval()

        X_imputed = tst.forward(X, pad_mask)

    X_imputed = X_imputed.numpy()
    y = y.flatten().numpy()

    n_folds = 5

    do_cv(X, y, n_splits=n_folds)

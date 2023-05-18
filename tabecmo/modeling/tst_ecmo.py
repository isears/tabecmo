import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabecmo.modeling.lookaheadTst import GenericPlTst, build_dl


def do_cv(ecmo_stayids: list, n_splits=5, base_model=None):
    cv = StratifiedKFold(n_splits=n_splits)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(ecmo_stayids)):
        train_dl = build_dl(ecmo_stayids[train_indices])
        test_dl = build_dl(ecmo_stayids[test_indices])

        if base_model:
            clf = GenericPlTst.load_from_checkpoint(base_model)
            # TODO: need to freeze appropriate layers here (and maybe add new classification head)

        else:
            clf = GenericPlTst()

        my_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="cache/ecmo_tst_models",
        )

        trainer = pl.Trainer(
            max_epochs=500,
            logger=False,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=True,
                    patience=10,
                    check_finite=False,
                ),
                my_checkpoint_callback,
            ],
            enable_checkpointing=True,
            default_root_dir="cache/tst_models",
            accelerator="gpu",
        )

        trainer.fit(model=clf, train_dataloaders=train_dl, val_dataloaders=test_dl)

        clf.load_from_checkpoint(my_checkpoint_callback.best_model_path)

    print(f"Average CV score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    ecmo_sids = studygroups[studygroups["ECMO"] == 1]["stay_id"].to_list()

    if len(sys.argv) > 1:
        print(f"Loading base model from {sys.argv[1]}")

        do_cv(ecmo_sids, base_model=sys.argv[1])

    else:
        print("Training without loading base model")
        do_cv(ecmo_sids, base_model=None)

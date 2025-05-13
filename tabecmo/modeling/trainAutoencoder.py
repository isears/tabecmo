import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split

from tabecmo.modeling.emrAutoencoder import EmrAutoencoder
import shutil

import argparse


def train_one_autoencoder(x_path: str, icu_name: str, pretraining_size: int):
    print(f"[*] Training {icu_name} autoencoder with input tensor: {x_path}")
    X_pretraining = torch.load(x_path).float()

    if pretraining_size > 0:
        indices = np.random.choice(
            range(0, X_pretraining.shape[0]), size=pretraining_size, replace=False
        )
        X_pretraining = X_pretraining[indices]

    print(X_pretraining.shape)

    clf = EmrAutoencoder()

    X_train, X_valid = train_test_split(X_pretraining, test_size=0.1, random_state=42)

    checkpointer = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=f"cache/best_{icu_name}_autoenc",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=3,
                check_finite=False,
            ),
            checkpointer,
        ],
        default_root_dir=f"cache/autoenc_{icu_name}",
        devices=[0],
    )

    trainer.fit(
        clf,
        train_dataloaders=torch.utils.data.TensorDataset(X_train),
        val_dataloaders=torch.utils.data.TensorDataset(X_valid),
    )

    shutil.copy(
        checkpointer.best_model_path,
        f"cache/saved_autoenc/{icu_name}.n{pretraining_size}.ckpt",
    )

    return checkpointer.best_model_path, checkpointer.best_model_score


def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "x_path",
    )

    parser.add_argument("-n", type=int, default=0, dest="n")

    return parser.parse_args()


if __name__ == "__main__":
    pl.seed_everything(42)

    args = argparse_setup()

    path_name_map = {
        "X_combined.pt": "combined",
        "X_Cardiac.Vascular.Intensive.Care.Unit.pt": "cvicu",
        "X_Coronary.Care.Unit.pt": "ccu",
        "X_Medical.Intensive.Care.Unit.pt": "micu",
        "X_Medical-Surgical.Intensive.Care.Unit.pt": "msicu",
        "X_Neuro.Intermediate.pt": "ni",
        "X_Neuro.Stepdown.pt": "ns",
        "X_Neuro.Surgical.Intensive.Care.Unit.pt": "nsicu",
        "X_Surgical.Intensive.Care.Unit.pt": "sicu",
        "X_Trauma.SICU.pt": "tsicu",
    }

    train_one_autoencoder(
        args.x_path, path_name_map[args.x_path.split("/")[-1]], args.n
    )

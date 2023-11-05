import shutil
from concurrent.futures import ProcessPoolExecutor, wait

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split

from tabecmo.modeling.emrAutoencoder import EmrAutoencoder
import argparse
import numpy as np

def train_one_autoencoder(args):
    x_path, icu_name, cline_args = args[0], args[1], args[2]
    print(f"[*] Training {icu_name} autoencoder with input tensor: {x_path}")
    X_pretraining = torch.load(f"cache/ihmtensors/{x_path}").float()

    if cline_args.n > 0:
        indices = np.random.choice(range(0, X_pretraining.shape[0]), size=cline_args.n, replace=False)
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
    )

    trainer.fit(
        clf,
        train_dataloaders=torch.utils.data.TensorDataset(X_train),
        val_dataloaders=torch.utils.data.TensorDataset(X_valid),
    )

    shutil.copy(checkpointer.best_model_path, f"cache/saved_autoenc/{icu_name}.n{cline_args.n}.ckpt")

    return checkpointer.best_model_path, checkpointer.best_model_score


def argparse_setup():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        type=int,
        default=0,
        dest='n'
    )

    return parser.parse_args()


if __name__ == "__main__":
    cline_args = argparse_setup()

    pl.seed_everything(42)
    data_root_path = "cache/ihmtensors"
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

    futures = list()

    with ProcessPoolExecutor(max_workers=5) as executor:
        args = [(k, v, cline_args) for k, v in path_name_map.items()]
        result = executor.map(train_one_autoencoder, args)

        for r in result:
            print(f"[+] {r}")

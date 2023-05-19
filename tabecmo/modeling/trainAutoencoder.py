import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split

from tabecmo.modeling.emrAutoencoder import EmrAutoencoder

if __name__ == "__main__":
    pl.seed_everything(42)
    x_path = sys.argv[1]
    X_pretraining = torch.load(x_path).float()

    clf = EmrAutoencoder()

    X_train, X_valid = train_test_split(X_pretraining, test_size=0.1, random_state=42)

    checkpointer = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="cache/best_autoenc",
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
        default_root_dir="cache/autoenc_models",
    )

    trainer.fit(
        clf,
        train_dataloaders=torch.utils.data.TensorDataset(X_train),
        val_dataloaders=torch.utils.data.TensorDataset(X_valid),
    )

    print(checkpointer.best_model_path)

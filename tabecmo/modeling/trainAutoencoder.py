import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabecmo.modeling.simpleFFNN import EmrAutoencoder, SimpleFFNN

if __name__ == "__main__":
    pl.seed_everything(42)
    # x_path = sys.argv[1]
    x_path = "cache/ihmtensors/X_Cardiac.Vascular.Intensive.Care.Unit.pt"
    X_pretraining = torch.load(x_path).float()
    y_pretraining = torch.load(x_path.replace("X", "y")).float()

    clf = EmrAutoencoder()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_pretraining, y_pretraining, test_size=0.1, random_state=42
    )

    checkpointer = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="cache/best_autoenc",
    )

    trainer = pl.Trainer(
        max_epochs=5,
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
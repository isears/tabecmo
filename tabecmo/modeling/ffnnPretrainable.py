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
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
)

pl.seed_everything(42)


class SimpleFFNN(pl.LightningModule):
    def __init__(self, n_features=89, hidden_dim=64, lr=1e-3) -> None:
        super().__init__()

        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.loss_fn = torch.nn.BCELoss()

        self.scorers = [
            BinaryAUROC().to("cuda"),
            BinaryPrecision().to("cuda"),
            BinaryF1Score().to("cuda"),
            BinaryAccuracy().to("cuda"),
        ]

        self.lr = lr

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        for s in self.scorers:
            s.update(preds=preds, target=y)

        self.log("val_loss", loss)

        return loss

    def on_validation_epoch_end(self):
        print("\n\nValidation scores:")

        for s in self.scorers:
            final_score = s.compute()
            print(f"\t{s.__class__.__name__}: {final_score}")
            self.log(f"Validation {s.__class__.__name__}", final_score)
            s.reset()

        print()

        return final_score

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.sigmoid(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def do_pretraining(pretrain_X, pretrain_y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        pretrain_X, pretrain_y, test_size=0.1, random_state=42
    )

    clf = SimpleFFNN()

    checkpointer = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="cache/best_ffnn",
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
        default_root_dir="cache/ffnn_models",
    )

    trainer.fit(
        clf,
        train_dataloaders=torch.utils.data.TensorDataset(
            X_train, y_train.unsqueeze(-1)
        ),
        val_dataloaders=torch.utils.data.TensorDataset(X_valid, y_valid.unsqueeze(-1)),
    )

    clf = clf.load_from_checkpoint(checkpointer.best_model_path)

    return clf


def do_cv(X, y, model, n_splits=5):
    print(X.shape)
    print(y.shape)
    cv = StratifiedKFold(n_splits=n_splits)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        model = SimpleFFNN()
        trainer = pl.Trainer(
            max_epochs=5,
            logger=False,
            enable_progress_bar=False,
            default_root_dir="cache/ffnn_models",
        )
        trainer.fit(
            model,
            train_dataloaders=torch.utils.data.TensorDataset(
                X[train_indices], y[train_indices].unsqueeze(-1)
            ),
        )

        with torch.no_grad():
            model = model.eval()
            preds = model.forward(X[test_indices])

        score = roc_auc_score(y[test_indices], preds)
        scores.append(score)
        print(f"Fold #{fold_idx} {score}")

    scores = np.array(scores)
    print(f"Average (std) CV score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()
    m = SimpleFFNN()
    do_cv(X_ecmo, y_ecmo, m)

    if len(sys.argv) > 1:
        print(f"Pretraining on {sys.argv[1]}")
        x_path = sys.argv[1]
        X_pretraining = torch.load(x_path).float()
        y_pretraining = torch.load(x_path.replace("X", "y")).float()
        base_model = do_pretraining(X_pretraining, y_pretraining)

        do_cv(X_ecmo, y_ecmo, base_model=SimpleFFNN())

    else:
        print(f"Training without pretraining")
        do_cv(X_ecmo, y_ecmo)

import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
)


class EmrAutoencoder(pl.LightningModule):
    def __init__(self, n_features=89, encoding_dim=16, lr=1e-3) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32), nn.ReLU(), nn.Linear(32, n_features)
        )
        self.loss_fn = F.mse_loss

        self.lr = lr

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        X = batch[0]
        preds = self.forward(X)

        # Give the model a "pass" on any values that were missing
        preds[X == -1] = -1
        loss = self.loss_fn(preds, X)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        preds = self.forward(X)

        # Give the model a "pass" on any values that were missing
        preds[X == -1] = -1
        loss = self.loss_fn(preds, X)

        # Pl doesn't do a great job of showing small losses
        self.log("val_loss", loss * 1e3)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class EncoderClassifier(pl.LightningModule):
    def __init__(self, autoencoder: EmrAutoencoder, lr=1e-3) -> None:
        super().__init__()

        self.encoder = autoencoder.encoder
        self.classification_head = nn.Linear(autoencoder.encoding_dim, 1)
        self.loss_fn = torch.nn.BCELoss()

        self.scorers = [BinaryAUROC(), BinaryPrecision()]

        self.lr = lr

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.classification_head(z)
        return torch.sigmoid(y_hat)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

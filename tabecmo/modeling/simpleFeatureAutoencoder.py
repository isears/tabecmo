import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import UnlabeledTimeseriesDataset


class SimpleFeatureAutoencoder(pl.LightningModule):
    def __init__(
        self,
        n_features,
        feat_encoding_dim,
        middle_encoding_dim,
        seq_len,
        lr,
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.feat_encoding_dim = feat_encoding_dim
        self.middle_encoding_dim = middle_encoding_dim
        self.lr = lr

        # TODO: if building individual per-feature encoders, need to use modulelist
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        # self.feature_encoders = list()
        # self.feature_decoders = list()

        # for fidx in range(0, n_features):
        #     self.feature_encoders.append(torch.nn.Linear(seq_len, feat_encoding_dim))
        #     self.feature_decoders.append(torch.nn.Linear(feat_encoding_dim, seq_len))
        self.feature_encoder = torch.nn.Linear(seq_len, feat_encoding_dim)
        self.feature_decoder = torch.nn.Linear(feat_encoding_dim, seq_len)

        self.middle_encoder = torch.nn.Linear(
            n_features * feat_encoding_dim, middle_encoding_dim
        )
        self.middle_decoder = torch.nn.Linear(
            middle_encoding_dim, n_features * feat_encoding_dim
        )

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

    def encode(self, X: torch.Tensor):
        encodings = torch.stack(
            [self.feature_encoder(X[:, :, fidx]) for fidx in range(0, self.n_features)],
            dim=-1,
        )

        encoded_features_flat = encodings.reshape(
            X.shape[0], self.n_features * self.feat_encoding_dim
        )

        return self.middle_encoder(encoded_features_flat)

    def decode(self, encoded: torch.Tensor):
        middle_decoded = self.middle_decoder(encoded).reshape(
            encoded.shape[0], self.n_features, self.feat_encoding_dim
        )

        decodings = torch.stack(
            [
                self.feature_decoder(middle_decoded[:, fidx, :])
                for fidx in range(0, self.n_features)
            ],
            dim=-1,
        )

        return decodings

    def forward(self, X: torch.Tensor, padding_mask: torch.Tensor):
        encoded = self.encode(X)
        decoded = self.decode(encoded)

        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)

        self.train_mse.update(preds=x_hat, target=x)

        return loss

    def on_train_epoch_end(self) -> None:
        print(f"\n\ntrain_loss: {self.train_mse.compute()}\n\n")
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        x, pm = batch
        x_hat = self.forward(x, pm)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("valid_los", loss)

        self.valid_mse.update(preds=x_hat, target=x)

        return loss

    def on_validation_epoch_end(self) -> None:
        print(f"\n\nval_loss: {self.valid_mse.compute()}\n\n")
        self.valid_mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def build_dl(stay_ids: list):
    ds = UnlabeledTimeseriesDataset(studygroups["stay_id"].to_list())

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
        collate_fn=ds.maxlen_padmask_collate,
        pin_memory=True,
    )

    return dl


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["ECMO"] == 0)
    ]

    studygroups = studygroups[studygroups["los"] < 30]

    train_sids, valid_sids = train_test_split(
        studygroups["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    train_dl = build_dl(train_sids)
    valid_dl = build_dl(valid_sids)

    # TODO: actually get input dim from dataset
    autoencoder = SimpleFeatureAutoencoder(
        n_features=89,
        feat_encoding_dim=10,
        middle_encoding_dim=50,
        seq_len=train_dl.dataset.max_len,
        lr=2e-3,
    )

    trainer = pl.Trainer(
        # limit_train_batches=100,
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
        ],
        default_root_dir="cache/encoder_models",
        accelerator="gpu",
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

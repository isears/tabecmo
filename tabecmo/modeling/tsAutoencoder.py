import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import UnlabeledTimeseriesDataset


class TimeseriesAutoencoder(pl.LightningModule):
    def __init__(
        self,
        feat_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dim_encoded,
        seq_len,
        lr,
    ):
        super().__init__()
        self.nhead = nhead

        self.linear = torch.nn.Linear(feat_dim, d_model)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=self.nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        )

        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        )

        self.linear_encoding = torch.nn.Linear(seq_len * d_model, dim_encoded)
        self.linear_decoding = torch.nn.Linear(dim_encoded, seq_len * d_model)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

        self.lr = lr
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, X: torch.Tensor, padding_masks: torch.Tensor):
        inp = X.permute(1, 0, 2)
        inp = self.linear(inp) * np.sqrt(self.d_model)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer

        encoded = self.encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        # the output transformer encoder/decoder embeddings don't include non-linearity

        # Output
        encoded = torch.nn.functional.gelu(encoded)
        # encoded = encoded.permute(1, 0, 2)
        # encoded = encoded * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        # encoded = encoded.reshape(
        #     encoded.shape[0], -1
        # )  # (batch_size, seq_length * d_model)

        # decoded = encoded.reshape(
        #     self.seq_len, encoded.shape[0], self.d_model
        # )  # (seq_len, batch_size, d_model)

        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss = torch.nn.functional.mse_loss(decoded, x)
        self.log("train_loss", loss)

        self.train_mse.update(preds=decoded, target=x)

        return loss

    def on_train_epoch_end(self) -> None:
        print(f"\n\ntrain_loss: {self.train_mse.compute()}\n\n")
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        x, pm = batch
        encoded = self.forward(x, pm)
        decoded = self.decoder(encoded)
        reconstructed = self.linear(decoded)
        loss = torch.nn.functional.mse_loss(reconstructed, x)
        self.log("val_loss", loss)

        self.valid_mse.update(preds=reconstructed, target=x)

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
        pin_memory=(config.gpus_available > 0),
        collate_fn=ds.maxlen_padmask_collate,
    )

    return dl


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["ECMO"] == 0)
        & (studygroups["los"] < 50)
    ]

    train_sids, valid_sids = train_test_split(
        studygroups["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    train_dl = build_dl(train_sids)
    valid_dl = build_dl(valid_sids)

    # TODO: actually get input dim from dataset
    autoencoder = TimeseriesAutoencoder(
        feat_dim=89,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dim_encoded=16,
        seq_len=train_dl.dataset.max_len,
        lr=1e-4,
        num_layers=2,
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
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=valid_dl)

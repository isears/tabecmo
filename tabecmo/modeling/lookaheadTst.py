import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import DerivedDataset
from tabecmo.dataProcessing.pretrainingDataset import ImputationDataset


class GenericPlTst(pl.LightningModule):
    def __init__(self, lr=1e-3) -> None:
        super().__init__()

        self.tst = TSTransformerEncoderClassiregressor(
            feat_dim=89,
            max_len=50 * 24,
            d_model=64,
            dim_feedforward=128,
            num_classes=89,
            num_layers=2,
            n_heads=8,
        )

        self.loss_fn = torch.nn.functional.mse_loss
        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

        self.lr = lr

    def training_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)

        # Only get loss at valid data points
        preds[y == -1] = -1
        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)

        preds[y == -1] = -1

        loss = self.loss_fn(preds, y)

        self.valid_mse.update(preds=preds, target=y)

        self.log("val_loss", loss)

        return loss

    def on_validation_epoch_end(self):
        print(f"\n\nval_loss: {self.valid_mse.compute()}\n\n")
        self.valid_mse.reset()

        print()

    def test_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)
        loss = self.loss_fn(preds, y)

        for s in self.scorers:
            s.update(preds, y.int())

        return loss

    def on_test_epoch_end(self):
        for s in self.scorers:
            final_score = s.compute()
            self.log(f"Test {s.__class__.__name__}", final_score)

        return final_score

    def forward(self, X, pad_masks):
        return self.tst(X, pad_masks)

    def forward_partial(self, X, pad_masks):
        inp = X.permute(1, 0, 2)
        inp = self.tst.project_inp(inp) * np.sqrt(self.tst.d_model)
        inp = self.tst.pos_enc(inp)
        output = self.tst.transformer_encoder(inp, src_key_padding_mask=~pad_masks)
        output = self.tst.act(output)
        output = output.permute(1, 0, 2)

        output = output * pad_masks.unsqueeze(-1)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def build_dl(stay_ids: list):
    ds = ImputationDataset(stay_ids)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=16,
        collate_fn=ds.maxlen_padmask_collate,
        pin_memory=True,
    )

    return dl


if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    studygroups = studygroups[
        (studygroups["unit_Cardiac Vascular Intensive Care Unit (CVICU)"] == 1)
        & (studygroups["ECMO"] == 0)
        & (studygroups["los"] < 50)
        & (studygroups["los"] > 2)
    ]

    train_sids, valid_sids = train_test_split(
        studygroups["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    train_dl = build_dl(train_sids)
    valid_dl = build_dl(valid_sids)

    # TODO: actually get input dim from dataset
    pl_tst = GenericPlTst()

    trainer = pl.Trainer(
        max_epochs=500,
        logger=False,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=3,
                check_finite=False,
            ),
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                dirpath="cache/best_tst_models",
            ),
        ],
        default_root_dir="cache/tst_models",
        enable_checkpointing=True,
        accelerator="gpu",
    )

    trainer.fit(
        model=pl_tst,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

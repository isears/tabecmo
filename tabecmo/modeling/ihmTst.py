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
    BinaryF1Score,
    BinaryPrecision,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import DerivedDataset, IhmLabelingDataset
from tabecmo.dataProcessing.pretrainingDataset import ImputationDataset


class GenericPlTst(pl.LightningModule):
    def __init__(self, lr=1e-3) -> None:
        super().__init__()

        self.tst = TSTransformerEncoderClassiregressor(
            feat_dim=89,
            max_len=50 * 24,
            d_model=64,
            dim_feedforward=128,
            num_classes=1,
            num_layers=2,
            n_heads=8,
        )

        self.loss_fn = torch.nn.BCELoss()

        self.scorers = [
            BinaryAUROC().to("cuda"),
            BinaryPrecision().to("cuda"),
            BinaryF1Score().to("cuda"),
        ]

        self.lr = lr

    def training_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)

        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)

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

    def forward(self, X, pad_masks):
        logits = self.tst(X, pad_masks)

        return torch.sigmoid(logits)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def build_dl(stay_ids: list, batch_size=16):
    ds = IhmLabelingDataset(stay_ids)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=batch_size,
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

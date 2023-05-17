import pandas as pd
import pytorch_lightning as pl
import torch
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from pytorch_lightning.callbacks import EarlyStopping
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


class GenericPlTst(pl.LightningModule):
    def __init__(self, tst, lr) -> None:
        super().__init__()
        self.tst = tst

        self.loss_fn = torch.nn.BCELoss()
        self.scorers = [
            MultilabelAUROC(num_labels=3).to("cuda"),
            MultilabelAveragePrecision(num_labels=3).to("cuda"),
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
            s.update(preds, y.int())

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

    def test_step(self, batch, batch_idx):
        X, y, pad_masks = batch
        preds = self.forward(X, pad_masks)
        loss = self.loss_fn(preds, y)

        for s in self.scorers:
            s.update(preds, y.int())

        return loss

    def on_test_epoch_end(self):
        test_competition_score = 0.0
        for s in self.scorers:
            final_score = s.compute()
            self.log(f"Test {s.__class__.__name__}", final_score)

        return test_competition_score

    def forward(self, X, pad_masks):
        logits = self.tst(X, pad_masks)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def build_dl(stay_ids: list):
    ds = DerivedDataset(stay_ids)

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

    studygroups = studygroups[(studygroups["los"] < 30) & (studygroups["los"] > 1)]

    train_sids, valid_sids = train_test_split(
        studygroups["stay_id"].to_list(), test_size=0.1, random_state=42
    )

    train_dl = build_dl(train_sids)
    valid_dl = build_dl(valid_sids)

    tst = TSTransformerEncoderClassiregressor(
        feat_dim=89,
        max_len=train_dl.dataset.max_len,
        d_model=64,
        dim_feedforward=128,
        num_classes=3,
        num_layers=1,
        n_heads=8,
    )

    # TODO: actually get input dim from dataset
    pl_tst = GenericPlTst(
        tst=tst,
        lr=1e-3,
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
        model=pl_tst,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

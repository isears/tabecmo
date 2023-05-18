import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from tabecmo import config
from tabecmo.dataProcessing.derivedDataset import IhmLabelingDataset
from tabecmo.modeling.ihmTst import GenericPlTst


def do_loo_fold(training_dl, leftout_dl, model: GenericPlTst):
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath="cache/ecmo_models",
    )

    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        callbacks=[
            checkpointer,
        ],
        enable_checkpointing=True,
        accelerator="gpu",
        enable_progress_bar=False,
    )

    for name, parameter in model.named_parameters():
        if not name.startswith("tst.output_layer"):
            parameter.requires_grad = False

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
    )

    model.load_from_checkpoint(checkpointer.best_model_path)

    model = model.eval()
    with torch.no_grad():
        X_leftout, y_leftout, pm = next(iter(leftout_dl))
        pred = model(X_leftout, pm)

    return pred, y_leftout


def generate_loo_data(stay_ids: list, stayid_leftout: int):
    training_sids = [s for s in stay_ids if s != stayid_leftout]

    def build_dl(ds_in):
        return torch.utils.data.DataLoader(
            ds_in,
            num_workers=config.cores_available,
            batch_size=16,
            collate_fn=ds.maxlen_padmask_collate,
            pin_memory=True,
        )

    ds = IhmLabelingDataset(training_sids)
    dl = build_dl(ds)

    ds_leftout = IhmLabelingDataset([stayid_leftout])
    dl_leftout = build_dl(ds_leftout)

    return dl, dl_leftout


if __name__ == "__main__":
    # Load stay ids
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    ecmo_stay_ids = studygroups[
        (studygroups["ECMO"] == 1)
        & (studygroups["los"] > 2)
        & (studygroups["los"] < 50)
    ]["stay_id"].to_list()

    all_preds = list()
    all_y = list()

    for stay_id in tqdm(ecmo_stay_ids):
        training_dl, leftout_dl = generate_loo_data(ecmo_stay_ids, stay_id)

        if len(sys.argv) > 1:
            print(f"Loading pretrained from {sys.argv[1]}")
            model = GenericPlTst.load_from_checkpoint(sys.argv[1])
        else:
            model = GenericPlTst()

        pred, y_actual = do_loo_fold(training_dl, leftout_dl, model)

        all_preds.append(pred.flatten().numpy()[0])
        all_y.append(y_actual.flatten().numpy()[0])

    final_score = roc_auc_score(all_y, all_preds)
    print(final_score)

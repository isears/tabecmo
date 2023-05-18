import sys

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from tabecmo.modeling.ihmTst import GenericPlTst, build_dl

if __name__ == "__main__":
    studygroups = pd.read_parquet("cache/studygroups.parquet")
    ecmo_stay_ids = studygroups[
        (studygroups["ECMO"] == 1)
        & (studygroups["los"] > 2)
        & (studygroups["los"] < 50)
    ]["stay_id"].to_list()

    dl = build_dl(ecmo_stay_ids, batch_size=len(ecmo_stay_ids))
    X, y, pm = next(iter(dl))

    model = GenericPlTst.load_from_checkpoint(sys.argv[1])
    model = model.eval()

    with torch.no_grad():
        preds = model(X, pm)

    print(roc_auc_score(y, preds))

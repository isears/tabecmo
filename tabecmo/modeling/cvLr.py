import copy

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# Need to re-write the functions from CV util b/c they're not designed to deal with an sklearn model
def do_one_fold(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train.unsqueeze(-1))

    preds = model.predict_proba(X_test)[:, 1]

    return y_test, preds


def do_loo_cv(X_cv, y_cv, model):
    all_ytrue = list()
    all_preds = list()

    for loo_idx in tqdm(range(0, X_cv.shape[0])):
        X_train = torch.cat((X_cv[:loo_idx], X_cv[loo_idx + 1 :]), dim=0)
        X_test = X_cv[loo_idx, :].unsqueeze(0)
        y_train = torch.cat((y_cv[:loo_idx], y_cv[loo_idx + 1 :]), dim=0)
        y_test = y_cv[loo_idx]

        y_true, pred = do_one_fold(X_train, X_test, y_train, y_test, model)
        all_ytrue.append(y_true.item())
        all_preds.append(pred.item())

    score = roc_auc_score(np.array(all_ytrue), np.array(all_preds))
    print(f"LOO CV final score: {score}")

    return score


if __name__ == "__main__":
    pl.seed_everything(42)
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    clf = LogisticRegression()
    score = do_loo_cv(X_ecmo, y_ecmo, clf)

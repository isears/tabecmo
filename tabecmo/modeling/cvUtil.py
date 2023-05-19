import copy

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def do_one_fold(X_train, X_test, y_train, y_test, model):
    this_fold_model = copy.deepcopy(model)
    trainer = pl.Trainer(
        max_epochs=3,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        this_fold_model,
        train_dataloaders=torch.utils.data.TensorDataset(
            X_train, y_train.unsqueeze(-1)
        ),
    )

    with torch.no_grad():
        this_fold_model = this_fold_model.eval()
        preds = this_fold_model.forward(X_test)

    return y_test, preds


def do_cv(X_cv, y_cv, model):
    cv = StratifiedKFold(n_splits=5)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X_cv, y_cv)):
        this_fold_y, this_fold_preds = do_one_fold(
            X_cv[train_indices],
            X_cv[test_indices],
            y_cv[train_indices],
            y_cv[test_indices],
            model,
        )
        score = roc_auc_score(this_fold_y, this_fold_preds)
        scores.append(score)

    scores = np.array(scores)
    print(f"Average (std) CV score: {np.mean(scores)} ({np.std(scores)})")

    return np.mean(scores)


def do_loo_cv(X_cv, y_cv, model):
    all_ytrue = list()
    all_preds = list()

    for loo_idx in tqdm(range(0, X_cv.shape[0])):
        X_train = torch.cat((X_cv[:loo_idx], X_cv[loo_idx + 1 :]), dim=0)
        X_test = X_cv[loo_idx, :].unsqueeze(0)
        y_train = torch.cat((y_cv[:loo_idx], y_cv[loo_idx + 1 :]), dim=0)
        y_test = torch.tensor(y_cv[loo_idx])

        y_true, pred = do_one_fold(X_train, X_test, y_train, y_test, model)
        all_ytrue.append(y_true.item())
        all_preds.append(pred.item())

    score = roc_auc_score(np.array(all_ytrue), np.array(all_preds))
    print(f"LOO CV final score: {score}")

    return score

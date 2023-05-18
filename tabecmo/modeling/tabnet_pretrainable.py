import sys

import torch
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def do_pretraining(unlabeled_data_path: str):
    X = torch.load(unlabeled_data_path).numpy()

    X_pretrain_train, X_pretrain_test = train_test_split(
        X, test_size=0.1, random_state=42
    )

    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
    )

    unsupervised_model.fit(
        X_train=X_pretrain_train,
        # eval_name=["validation"],
        # eval_metric=["mse"],
        # eval_set=[X_pretrain_test],
        pretraining_ratio=0.25,
        patience=3,
        max_epochs=3,
    )

    return unsupervised_model


def do_cv(X, y, n_splits=5, base_model=None):
    cv = StratifiedKFold(n_splits=n_splits)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            # scheduler_params={
            #     "step_size": 10,  # how to use learning rate scheduler
            #     "gamma": 0.9,
            # },
            # scheduler_fn=torch.optim.lr_scheduler.StepLR,
            # mask_type="sparsemax",  # This will be overwritten if using pretrain model
        )

        if base_model:
            clf.fit(
                X_train=X[train_indices],
                y_train=y[train_indices],
                eval_set=[
                    (X[train_indices], y[train_indices]),
                    (X[test_indices], y[test_indices]),
                ],
                eval_name=["train", "test"],
                eval_metric=["auc"],
                from_unsupervised=base_model,
            )
        else:
            clf.fit(
                X[train_indices],
                y[train_indices],
                eval_set=[(X[test_indices], y[test_indices])],
                eval_name=["test"],
                eval_metric=["auc"],
            )

        preds = clf.predict_proba(X[test_indices])[:, 1]
        score = roc_auc_score(y[test_indices], preds)
        scores.append(score)
        print(f"Fold #{fold_idx} {score}")

    print(f"Average CV score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").numpy()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").numpy()

    X_ecmo_train, X_ecmo_test, y_ecmo_train, y_ecmo_test = train_test_split(
        X_ecmo, y_ecmo, test_size=0.2, random_state=42
    )

    # TODO: debug only
    # sys.argv.append("cache/X_unlabeled_Cardiac.Vascular.Intensive.Care.Unit.pt")
    if len(sys.argv) > 1:
        print(f"Training unsupervised on {sys.argv[1]}")
        base_model = do_pretraining(sys.argv[1])

        do_cv(X_ecmo, y_ecmo, base_model=base_model)

    else:
        print(f"Training without unsupervised pretraining")
        do_cv(X_ecmo, y_ecmo)

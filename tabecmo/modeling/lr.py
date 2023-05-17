import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def do_cv(X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits)
    scores = list()

    for fold_idx, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        lr = LogisticRegression()
        lr.fit(
            X[train_indices],
            y[train_indices],
        )

        preds = lr.predict_proba(X[test_indices])[:, 1]
        score = roc_auc_score(y[test_indices], preds)
        scores.append(score)
        print(f"Fold #{fold_idx} {score}")

    scores = np.array(scores)
    print(f"Average (std) CV score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    X_ecmo = torch.load("cache/X_ecmo.pt").numpy()
    y_ecmo = torch.load("cache/y_ecmo.pt").numpy()

    for label_idx, label in enumerate(["thrombosis", "hemorrhage", "stroke"]):
        print(f"Running CV for target {label}")
        do_cv(X_ecmo, y_ecmo[:, label_idx])

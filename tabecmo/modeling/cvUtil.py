from sklearn.model_selection import KFold, StratifiedKFold


def generate_folds(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits)

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx])

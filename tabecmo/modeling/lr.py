import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X_ecmo = torch.load("cache/X_ecmo.pt").numpy()
    y_ecmo = torch.load("cache/y_ecmo.pt").numpy()

    X_ecmo_train, X_ecmo_test, y_ecmo_train, y_ecmo_test = train_test_split(
        X_ecmo, y_ecmo, test_size=0.2, random_state=42
    )

    lr = LogisticRegression()
    lr.fit(X_ecmo_train, y_ecmo_train[:, 0])

    preds = lr.predict_proba(X_ecmo_test)[:, 1]

    print(roc_auc_score(y_ecmo_test[:, 0], preds))

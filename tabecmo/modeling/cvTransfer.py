import numpy as np
import pytorch_lightning as pl
import torch

from tabecmo.modeling.cvUtil import do_cv, do_loo_cv
from tabecmo.modeling.simpleFFNN import SimpleFFNN

if __name__ == "__main__":
    pl.seed_everything(42)
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    clf = SimpleFFNN.load_from_checkpoint(
        "/home/isears/Repos/tabnet-ecmo/cache/best_ffnn/epoch=4-step=25290-v3.ckpt",
        # lr=1e-4,
    )

    print(clf.lr)

    # for name, parameter in clf.named_parameters():
    #     if not name.startswith("fc2"):
    #         parameter.requires_grad = False

    do_loo_cv(X_ecmo, y_ecmo[:, 2], clf)

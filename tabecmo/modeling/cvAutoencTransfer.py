import numpy as np
import pytorch_lightning as pl
import torch

from tabecmo.modeling.cvUtil import do_cv, do_loo_cv
from tabecmo.modeling.simpleFFNN import EmrAutoencoder, EncoderClassifier, SimpleFFNN

if __name__ == "__main__":
    pl.seed_everything(42)
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    autoencoder = EmrAutoencoder.load_from_checkpoint(
        "/home/isears/Repos/tabnet-ecmo/cache/best_autoenc/epoch=4-step=20615-v2.ckpt",
        # lr=1e-4,
    )

    clf = EncoderClassifier(autoencoder)

    # for name, parameter in clf.named_parameters():
    #     if not name.startswith("fc2"):
    #         parameter.requires_grad = False

    do_cv(X_ecmo, y_ecmo, clf)

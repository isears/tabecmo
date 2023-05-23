import sys

import numpy as np
import pytorch_lightning as pl
import torch

from tabecmo.modeling.cvUtil import do_cv, do_loo_cv
from tabecmo.modeling.emrAutoencoder import EmrAutoencoder, EncoderClassifier

if __name__ == "__main__":
    pl.seed_everything(42)
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    autoencoder = EmrAutoencoder.load_from_checkpoint(sys.argv[1])

    clf = EncoderClassifier(autoencoder, lr=1e-3)

    do_loo_cv(X_ecmo, y_ecmo, clf)

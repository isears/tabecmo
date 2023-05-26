import logging
import sys
from concurrent.futures import ProcessPoolExecutor

import pytorch_lightning as pl
import torch

from tabecmo.modeling.cvUtil import do_cv, do_loo_cv
from tabecmo.modeling.emrAutoencoder import EmrAutoencoder, EncoderClassifier


def cv_one_model(model_name):
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    if model_name is not None:
        autoencoder = EmrAutoencoder.load_from_checkpoint(
            f"cache/saved_autoenc/{model_name}.ckpt"
        )
        clf = EncoderClassifier(autoencoder)
    else:
        autoencoder = EmrAutoencoder()
        clf = EncoderClassifier(autoencoder)

    return do_loo_cv(X_ecmo, y_ecmo, clf)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pl.seed_everything(int(sys.argv[1]))
    else:
        pl.seed_everything(42)

    logging.getLogger("lightning").setLevel(logging.ERROR)

    data_root_path = "cache/ihmtensors"
    available_autoencoders = {
        "X_Cardiac.Vascular.Intensive.Care.Unit.pt": "cvicu",
        "X_Coronary.Care.Unit.pt": "ccu",
        "X_Medical.Intensive.Care.Unit.pt": "micu",
        "X_Medical-Surgical.Intensive.Care.Unit.pt": "msicu",
        "X_Neuro.Intermediate.pt": "ni",
        "X_Neuro.Stepdown.pt": "ns",
        "X_Neuro.Surgical.Intensive.Care.Unit.pt": "nsicu",
        "X_Surgical.Intensive.Care.Unit.pt": "sicu",
        "X_Trauma.SICU.pt": "tsicu",
        "X_combined.pt": "combined",
    }

    futures = list()

    with ProcessPoolExecutor(max_workers=5) as executor:
        args = [(v) for _, v in available_autoencoders.items()]
        args += [
            None
        ]  # last model (ECMO w/out pretraining) will not load a saved autoencoder
        result = executor.map(cv_one_model, args)

        executor.shutdown(wait=True, cancel_futures=False)

    print("[+] Done with evaluations")
    print(15 * "=")
    for model_name, score in zip(args, result):
        if model_name is None:
            model_name = "unpretrained"

        print(f"{model_name}: {score}")

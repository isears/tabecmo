import logging
import sys
from concurrent.futures import ProcessPoolExecutor

import pytorch_lightning as pl
import torch

from tabecmo.modeling.cvUtil import do_cv, do_loo_cv
from tabecmo.modeling.emrAutoencoder import EmrAutoencoder, EncoderClassifier
import argparse
import numpy as np
import datetime
import pandas as pd
import os

def cv_one_model(args):
    model_name, pretraining_size, fine_tuning_size = args
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    if fine_tuning_size > 0:
        indices = np.random.choice(range(0, X_ecmo.shape[0]), size=fine_tuning_size, replace=False)
        X_ecmo = X_ecmo[indices]
        y_ecmo = y_ecmo[indices]

    if model_name is not None:
        autoencoder = EmrAutoencoder.load_from_checkpoint(
            f"cache/saved_autoenc/{model_name}.n{pretraining_size}.ckpt"
        )
        clf = EncoderClassifier(autoencoder)
    else:
        autoencoder = EmrAutoencoder()
        clf = EncoderClassifier(autoencoder)

    return do_loo_cv(X_ecmo, y_ecmo, clf)


def argparse_setup():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        type=int,
        default=0,
        dest='n',
        help='pretraining size (requires existance of pretrained model @ n)'
    )

    parser.add_argument(
        '-s',
        type=int,
        default=42,
        dest='s',
        help='random seed'
    )

    parser.add_argument(
        '-m',
        type=int,
        default=0,
        dest='m',
        help='fine tuning size'
    )

    return parser.parse_args()

if __name__ == "__main__":
    cline_args = argparse_setup()
    pl.seed_everything(cline_args.s)


    logging.getLogger("lightning").setLevel(logging.ERROR)

    data_root_path = "cache/ihmtensors"
    available_autoencoders = {
        "X_Cardiac.Vascular.Intensive.Care.Unit.pt": "cvicu",
        "X_Coronary.Care.Unit.pt": "ccu",
        "X_Medical.Intensive.Care.Unit.pt": "micu",
        "X_Medical-Surgical.Intensive.Care.Unit.pt": "msicu",
        # These don't have enough examples to support 1k minimum pretraining
        # "X_Neuro.Intermediate.pt": "ni",
        # "X_Neuro.Stepdown.pt": "ns",
        "X_Neuro.Surgical.Intensive.Care.Unit.pt": "nsicu",
        "X_Surgical.Intensive.Care.Unit.pt": "sicu",
        "X_Trauma.SICU.pt": "tsicu",
        "X_combined.pt": "combined",
    }

    futures = list()

    with ProcessPoolExecutor(max_workers=5) as executor:
        args = [(v, cline_args.n, cline_args.m) for _, v in available_autoencoders.items()]
        args += [
            (None, cline_args.n, cline_args.m)
        ]  # last model (ECMO w/out pretraining) will not load a saved autoencoder
        result = executor.map(cv_one_model, args)

        executor.shutdown(wait=True, cancel_futures=False)

    print("[+] Done with evaluations")
    print(15 * "=")

    saved_results = list()
    for (model_name, _, _), score in zip(args, result):
        print(f"{model_name}: {score}")
        saved_results.append(score)


    this_run_df = pd.DataFrame(data={
        'Pretraining': [model_name for (model_name, _, _) in args],
        'Pretraining Size': [cline_args.n] * len(args),
        'Fine Tuning Size': [cline_args.m] * len(args),
        'Score': saved_results,
        'Seed': [cline_args.s] * len(args),
        'Timestamp': [datetime.datetime.now()] * len(args)
    })

    if not os.path.exists('results/performance.parquet'):
        results_df = pd.DataFrame()
    else:
        results_df = pd.read_parquet('results/performance.parquet')

    updated_df = pd.concat([results_df, this_run_df], ignore_index=True)
    updated_df = updated_df.sort_values('Timestamp').drop_duplicates(subset=[c for c in updated_df.columns if c not in ['Timestamp', 'Score']], keep='last').reset_index(drop=True)
    updated_df.to_parquet('results/performance.parquet')

        


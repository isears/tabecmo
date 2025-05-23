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
        indices = np.random.choice(
            range(0, X_ecmo.shape[0]), size=fine_tuning_size, replace=False
        )
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

    parser.add_argument("-s", type=int, default=42, dest="s", help="random seed")

    # parser.add_argument("-m", type=int, default=0, dest="m", help="fine tuning size")

    return parser.parse_args()


if __name__ == "__main__":
    cline_args = argparse_setup()
    pl.seed_everything(cline_args.s)

    logging.getLogger("lightning").setLevel(logging.ERROR)

    futures = list()

    # TODO: hard-coded for now
    available_pretraining_sizes = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]
    model_names = ["cvicu", "micu", "msicu", "ni", "nsicu", "ns", "sicu", "tsicu"]

    with ProcessPoolExecutor(max_workers=5) as executor:
        # all icus pretrained model, various pretraining dataset sizes, finetune on entire ecmo dataset
        args = [("combined", n, 0) for n in available_pretraining_sizes]
        # last model (ECMO w/out pretraining) will not load a saved autoencoder
        args += [(None, 0, 0)]
        # test various icu-specific pretrained models
        args += [(n, 0, 0) for n in model_names]
        # test various icu-specific pretrained models with fixed pretraining size
        args += [(n, 1000, 0) for n in model_names if n not in ["ns", "nsicu"]]
        # test full all icus pretrained model with various finetuning sizes
        args += [
            ("combined", 0, i)
            for i in [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 0]
        ]
        # test micu pretrained model with various finetuning sizes
        args += [
            ("micu", 0, i) for i in [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 0]
        ]

        result = executor.map(cv_one_model, args)

        executor.shutdown(wait=True, cancel_futures=False)

    print("[+] Done with evaluations")
    print(15 * "=")

    saved_results = list()
    for (model_name, pretraining_size, fine_tuning_size), score in zip(args, result):
        print(f"{model_name}.{pretraining_size}.{fine_tuning_size}: {score}")
        saved_results.append(score)

    this_run_df = pd.DataFrame(
        data={
            "Pretraining": [model_name for (model_name, _, _) in args],
            "Pretraining Size": [n for (_, n, _) in args],
            "Fine Tuning Size": [n for (_, _, n) in args],
            "Score": saved_results,
            "Seed": [cline_args.s] * len(args),
            "Timestamp": [datetime.datetime.now()] * len(args),
        }
    )

    if not os.path.exists("results/performance.parquet"):
        results_df = pd.DataFrame()
    else:
        results_df = pd.read_parquet("results/performance.parquet")

    updated_df = pd.concat([results_df, this_run_df], ignore_index=True)
    updated_df = (
        updated_df.sort_values("Timestamp")
        .drop_duplicates(
            subset=[c for c in updated_df.columns if c not in ["Timestamp", "Score"]],
            keep="last",
        )
        .reset_index(drop=True)
    )
    updated_df.to_parquet("results/performance.parquet")

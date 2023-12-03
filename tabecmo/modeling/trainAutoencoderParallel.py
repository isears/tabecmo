from concurrent.futures import ProcessPoolExecutor

import pytorch_lightning as pl


from tabecmo.modeling.trainAutoencoder import train_one_autoencoder
import argparse


def train_one_autoencoder_wrapper(args):
    x_path, icu_name, cline_args = args[0], args[1], args[2]
    pretraining_size = cline_args.n

    train_one_autoencoder(x_path, icu_name, pretraining_size)



def argparse_setup():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        type=int,
        default=0,
        dest='n'
    )

    return parser.parse_args()


if __name__ == "__main__":
    cline_args = argparse_setup()

    pl.seed_everything(42)
    data_root_path = "cache/ihmtensors"
    path_name_map = {
        "X_combined.pt": "combined",
        "X_Cardiac.Vascular.Intensive.Care.Unit.pt": "cvicu",
        "X_Coronary.Care.Unit.pt": "ccu",
        "X_Medical.Intensive.Care.Unit.pt": "micu",
        "X_Medical-Surgical.Intensive.Care.Unit.pt": "msicu",
        "X_Neuro.Intermediate.pt": "ni",
        "X_Neuro.Stepdown.pt": "ns",
        "X_Neuro.Surgical.Intensive.Care.Unit.pt": "nsicu",
        "X_Surgical.Intensive.Care.Unit.pt": "sicu",
        "X_Trauma.SICU.pt": "tsicu",
    }

    futures = list()

    with ProcessPoolExecutor(max_workers=5) as executor:
        args = [(f"{data_root_path}/{k}", v, cline_args) for k, v in path_name_map.items()]
        result = executor.map(train_one_autoencoder, args)

        for r in result:
            print(f"[+] {r}")

from captum.attr import IntegratedGradients
import numpy as np
import pytorch_lightning as pl
import torch
import tempfile
from sklearn.metrics import roc_auc_score

from tabecmo.modeling.emrAutoencoder import EmrAutoencoder, EncoderClassifier

if __name__ == "__main__":
    pl.seed_everything(42)
    X_ecmo = torch.load("cache/ihmtensors/X_ecmo.pt").float()
    y_ecmo = torch.load("cache/ihmtensors/y_ecmo.pt").float()

    autoencoder = EmrAutoencoder.load_from_checkpoint("cache/saved_autoenc/combined.n0.ckpt")
    clf = EncoderClassifier(autoencoder)

    temp_dir_model = tempfile.TemporaryDirectory()

    trainer = pl.Trainer(
        max_epochs=7,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=temp_dir_model.name,
    )
    trainer.fit(
        clf,
        train_dataloaders=torch.utils.data.TensorDataset(
            X_ecmo, y_ecmo.unsqueeze(-1)
        ),
    )

    with torch.no_grad():
        clf_eval = clf.eval()
        preds = clf_eval.forward(X_ecmo)

        print(f"[*] Sanity check: {roc_auc_score(y_ecmo, preds)}")

        ig = IntegratedGradients(clf_eval)
        baseline = torch.zeros_like(X_ecmo)
        attributions = ig.attribute(X_ecmo, baseline, target=0, return_convergence_delta=False)
        print(attributions)
        print(attributions.shape)
        torch.save(attributions, "cache/attribs.pt")

    


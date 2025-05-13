#!/bin/bash

# for pretraining_size in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
# do
#     python tabecmo/modeling/trainAutoencoder.py cache/ihmtensors/X_combined.pt -n $pretraining_size
# done

for pretrain_unit in "X_Cardiac.Vascular.Intensive.Care.Unit.pt" "X_Coronary.Care.Unit.pt" "X_Medical.Intensive.Care.Unit.pt" "X_Medical-Surgical.Intensive.Care.Unit.pt" "X_Neuro.Intermediate.pt" "X_Neuro.Stepdown.pt" "X_Neuro.Surgical.Intensive.Care.Unit.pt" "X_Surgical.Intensive.Care.Unit.pt" "X_Trauma.SICU.pt"
do
    python tabecmo/modeling/trainAutoencoder.py cache/ihmtensors/$pretrain_unit -n 1000
done
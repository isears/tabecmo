#!/bin/bash

for pretraining_size in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
    python tabecmo/modeling/trainAutoencoder.py cache/ihmtensors/X_combined.pt -n $pretraining_size
done
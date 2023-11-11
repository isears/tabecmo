#!/bin/bash

for seed in 42 1 2 3 4
do
    # Moder performance by finetuning dataset size
    for finetuning_size in 20 30 40 50 60 70
    do
        python tabecmo/modeling/cvParallel.py -m $finetuning_size -s $seed
    done

    # Model performance by pretraining dataset size (set vs max available)
    python tabecmo/modeling/cvParallel.py -s $seed -n 0
    python tabecmo/modeling/cvParallel.py -s $seed -n 1000

done

#!/bin/bash

for seed in 42 1 2 3 4
do
    python tabecmo/modeling/cvParallel.py -s $seed

done

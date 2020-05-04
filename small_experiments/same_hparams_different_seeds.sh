#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

export PYTHONPATH="~/projects/dctn"
export CUDA_VISIBLE_DEVICES=0

for seed in 2 3 4 5 6 7; do
  python ~/projects/dctn/new_runner.py \
         --seed $seed \
         --ds-type fashionmnist \
         --ds-path /mnt/hdd_1tb/datasets/fashionmnist \
         --batch-size 128 \
         --no-breakpoint-on-nan-loss \
         --epses-specs '(4,4),(3,6)' \
         --no-es-train-acc \
         --no-es-train-mean-ce \
         --eval-schedule '((10,1),(100,10),(1000,100),(10000,1000),(None,2000))' \
         --patience 50 \
         --experiments-dir /mnt/important/experiments/2_epses_plus_linear_fashionmnist/2020-05-04_same_hparams_different_seeds \
         --reg-coeff 1e-2 \
         --reg-type epses_composition \
         --optimizer adam \
         --lr 1.821e-4 \
         --init-epses-composition-unit-empirical-output-std
done

#!/bin/bash
#SBATCH --job-name=flask_demo
#SBATCH --partition=vision
#SBATCH --cpus-per-task=7
#SBATCH --mem=8000
#SBATCH --nodelist=orange5

python flask_main.py \
    --net_file=../../scripts/imagenet.decafnet.epoch90 \
    --meta_file=../../scripts/imagenet.decafnet.meta

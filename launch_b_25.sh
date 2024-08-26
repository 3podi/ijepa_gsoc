#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q preempt

#SBATCH -t 24:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=2

#SBATCH --requeue 


python3 main_iris.py --fname configs/vit_b_7_25_iris.yaml --devices cuda:0 &
python3 main_iris.py --fname configs/vit_b_9_25_iris.yaml --devices cuda:1

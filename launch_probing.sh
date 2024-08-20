#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q preempt

#SBATCH -t 24:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=1

#SBATCH --requeue 


timeout 23h python3 linear_probing.py --fname configs_probing/probing_b_14.yaml --devices cuda:0

wait

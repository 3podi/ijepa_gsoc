#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q preempt

#SBATCH -t 24:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=3

#SBATCH --requeue 

python3 linear_probing.py --fname configs_probing/probing_s_14_75.yaml --devices cuda:0 &
python3 linear_probing.py --fname configs_probing/probing_s_9_75.yaml --devices cuda:1 &
python3 linear_probing.py --fname configs_probing/probing_s_7_75.yaml --devices cuda:2
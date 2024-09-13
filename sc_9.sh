#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q regular

#SBATCH -t 4:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=3

python3 linear_probing.py --fname generated_configs/probing_vit_s_9_25_1e4_fix_False_0.001.yaml --devices cuda:0 &
python3 linear_probing.py --fname generated_configs/probing_vit_s_14_75_1e4_fix_True_0.1.yaml --devices cuda:1 &
python3 linear_probing.py --fname generated_configs/probing_vit_b_14_25_1e4_fix_True_0.1.yaml --devices cuda:2




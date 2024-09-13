#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q regular

#SBATCH -t 24:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=4

python3 linear_probing.py --fname generated_configs_scratch/probing_vit_s_14_scratch_False_0.1.yaml --devices cuda:0 &
python3 linear_probing.py --fname generated_configs_scratch/probing_resnet_50_scratch_0.001.yaml --devices cuda:1 &
python3 linear_probing.py --fname generated_configs_scratch/probing_vit_s_7_scratch_True_0.0001.yaml --devices cuda:2 &
python3 linear_probing.py --fname generated_configs_scratch/probing_vit_b_9_scratch_False_0.0001.yaml --devices cuda:3




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

#SBATCH --requeue 

python3 main_iris.py --fname configs/vit_b_9_75_1e5_fix.yaml --devices cuda:0 &
python3 main_iris.py --fname configs/vit_b_9_75_1e5.yaml --devices cuda:1 &
python3 main_iris.py --fname configs/vit_b_9_75_1e4_fix.yaml --devices cuda:2 &
python3 main_iris.py --fname configs/vit_b_9_75_1e5_2.yaml --devices cuda:3

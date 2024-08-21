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


timeout 23h python3 main_iris.py --fname configs/vit_s_14_25_exp_iris.yaml --devices cuda:0 &
timeout 23h python3 main_iris.py --fname configs/vit_s_9_25_exp_iris.yaml --devices cuda:1 &
timeout 23h python3 main_iris.py --fname configs/vit_s_7_25_exp_iris.yaml --devices cuda:2

wait

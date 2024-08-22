#!/bin/bash

#SBATCH -A m4392

#SBATCH -C gpu

#SBATCH -q preempt

#SBATCH -t 12:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --gpus-per-task=1

#SBATCH --requeue 


python3 main_iris.py --fname configs_probing/vit_s_7_25_iris.yaml --devices cuda:0

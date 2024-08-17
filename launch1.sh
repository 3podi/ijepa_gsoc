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

start=
(
(
(((date +%H)+
(
(
(((date +%j)*24))))

timeout 23h python3 main_iris.py --fname configs/vit_b_14_25_iris.yaml --devices cuda:0 &
timeout 23h python3 main_iris.py --fname configs/vit_b_14_50_iris.yaml --devices cuda:0 &
timeout 23h python3 main_iris.py --fname configs/vit_b_14_75_iris.yaml --devices cuda:0 &

wait

end=
(
(
(((date +%H)+
(
(
(((date +%j)*24))))

tot=$((end-start))

if [ $tot -gt 1 ]; then
    sbatch ./Scripts/{script_name}

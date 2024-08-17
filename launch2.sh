#!/bin/bash

#SBATCH -C cpu

#SBATCH -q preempt

#SBATCH -t 1:00:00

#SBATCH -n 1

#SBATCH --ntasks-per-node=1

#SBATCH -c 128

#SBATCH --mem=0

#SBATCH --requeue 

start=
(
(
(((date +%H)+
(
(
(((date +%j)*24))))

timeout 23h python3 prova.py

wait

end=
(
(
(((date +%H)+
(
(
(((date +%j)*24))))

tot=$((end-start))


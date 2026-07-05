#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_dense_rRT_cont_0_8.%J.err 

#SBATCH --output=slurm_dense_rRT_cont_0_8.%J.out

python3 dense_loss_rRT_cont.py 0.8 100000



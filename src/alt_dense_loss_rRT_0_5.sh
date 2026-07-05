#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_alt_dense_rRT_0_5.%J.err 

#SBATCH --output=slurm_alt_dense_rRT_0_5.%J.out

python3 alt_dense_loss_rRT.py 0.5 50000



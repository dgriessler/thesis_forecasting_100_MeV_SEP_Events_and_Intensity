#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=5GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_alt_dense_0_9_cont.%J.err 

#SBATCH --output=slurm_alt_dense_0_9_cont.%J.out

python3 alt_dense_loss_cont.py 0.9 50000



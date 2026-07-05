#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=5GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_dense_0_1_cont.%J.err 

#SBATCH --output=slurm_dense_0_1_cont.%J.out

python3 dense_loss_cont.py 0.1 50000



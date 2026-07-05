#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=5GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_dense_1_5.%J.err 

#SBATCH --output=slurm_dense_1_5.%J.out

python3 dense_loss.py 1.5 50000



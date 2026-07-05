#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_auto_diff_lr.%J.err 

#SBATCH --output=slurm_auto_diff_lr.%J.out

python3 autoencoder_diff_lr.py



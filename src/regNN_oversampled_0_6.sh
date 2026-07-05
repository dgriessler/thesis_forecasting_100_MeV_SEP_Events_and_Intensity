#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=4

#SBATCH --error=slurm_reg_oversampled_0_6.%J.err 

#SBATCH --output=slurm_reg_oversampled_0_6.%J.out

python3 regNN_oversampled.py 6 15000



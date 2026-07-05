#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_reg_oversampled_0_9_2.%J.err 

#SBATCH --output=slurm_reg_oversampled_0_9_2.%J.out

python3 retrained_regNN_oversampled_single.py 9 39929 2



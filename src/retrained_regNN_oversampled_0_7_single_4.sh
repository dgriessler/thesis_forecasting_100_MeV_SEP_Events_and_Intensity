#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=10GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=4

#SBATCH --error=slurm_reg_oversampled_0_7_4.%J.err 

#SBATCH --output=slurm_reg_oversampled_0_7_4.%J.out

python3 retrained_regNN_oversampled_single.py 7 21550 4



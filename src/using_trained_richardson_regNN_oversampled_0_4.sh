#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_utr_reg_oversampled_0_4.%J.err 

#SBATCH --output=slurm_utr_reg_oversampled_0_4.%J.out

python3 using_trained_richardson_regNN_oversampled.py 4 35000



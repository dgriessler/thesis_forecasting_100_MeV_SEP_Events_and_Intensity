#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_rm_reg_retrained.%J.err 

#SBATCH --output=slurm_rm_reg_retrained.%J.out

python3 richardson_mixed_retrained_regNN.py



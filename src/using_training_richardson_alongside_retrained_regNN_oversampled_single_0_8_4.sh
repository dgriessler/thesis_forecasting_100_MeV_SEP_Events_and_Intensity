#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_utr_along_retrained_reg_oversampled_0_8_4.%J.err 

#SBATCH --output=slurm_utr_along_retrained_reg_oversampled_0_8_4.%J.out

python3 using_training_richardson_alongside_retrained_regNN_oversampled_single.py 8 46344 4



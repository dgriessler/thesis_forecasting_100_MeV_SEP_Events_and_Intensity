#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=20GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_reg_oversampled_classifier_0_8.%J.err 

#SBATCH --output=slurm_reg_oversampled_classifier_0_8.%J.out

python3 regNN_oversampled_classifier.py 8 15000



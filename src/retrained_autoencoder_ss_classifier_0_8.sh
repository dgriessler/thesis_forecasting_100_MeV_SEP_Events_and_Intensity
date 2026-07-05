#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_auto_ss_classifier_retrained_0_8.%J.err 

#SBATCH --output=slurm_auto_ss_classifier_retrained_0_8.%J.out

python3 retrained_autoencoder_ss_classifier.py 8 263



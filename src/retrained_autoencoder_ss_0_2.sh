#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_auto_ss_0_2_retrained.%J.err 

#SBATCH --output=slurm_auto_ss_0_2_retrained.%J.out

python3 retrained_autoencoder_ss.py 2 295



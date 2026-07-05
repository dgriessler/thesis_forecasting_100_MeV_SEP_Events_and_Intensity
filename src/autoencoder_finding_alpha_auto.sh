#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_auto_find_alpha_auto.%J.err 

#SBATCH --output=slurm_auto_find_alpha_auto.%J.out

python3 autoencoder_finding_alpha_auto.py



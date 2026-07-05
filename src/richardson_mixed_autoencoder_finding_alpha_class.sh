#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_rm_auto_finding_alpha_class.%J.err 

#SBATCH --output=slurm_rm_auto_finding_alpha_class.%J.out

python3 richardson_mixed_autoencoder_finding_alpha_class.py



#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_rm_auto_ss_0_4.%J.err 

#SBATCH --output=slurm_rm_auto_ss_0_4.%J.out

python3 richardson_mixed_autoencoder_ss.py 4 20000



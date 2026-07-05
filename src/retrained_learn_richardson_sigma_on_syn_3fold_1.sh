#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_retrained_learn_richardson_sigma_on_syn_3fold_1.%J.err 

#SBATCH --output=slurm_retrained_learn_richardson_sigma_on_syn_3fold_1.%J.out

python3 retrained_learn_richardson_sigma_on_syn_3fold.py 1 19863 0.0001



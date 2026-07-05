#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_learn_richardson_sigma_on_syn_3fold_0_3.%J.err 

#SBATCH --output=slurm_learn_richardson_sigma_on_syn_3fold_0_3.%J.out

python3 learn_richardson_sigma_on_syn_3fold.py 0 3



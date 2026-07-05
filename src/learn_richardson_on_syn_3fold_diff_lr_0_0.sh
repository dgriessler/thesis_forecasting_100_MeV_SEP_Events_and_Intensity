#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=10GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_learn_richardson_on_syn_3fold_diff_lr_0_0.%J.err 

#SBATCH --output=slurm_learn_richardson_on_syn_3fold_diff_lr_0_0.%J.out

python3 learn_richardson_on_syn_3fold_diff_lr.py 0 0



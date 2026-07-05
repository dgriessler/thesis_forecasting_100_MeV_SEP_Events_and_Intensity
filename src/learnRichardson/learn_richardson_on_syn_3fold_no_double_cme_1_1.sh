#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=10GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_learn_richardson_on_syn_3fold_no_double_cme_1_1.%J.err 

#SBATCH --output=slurm_learn_richardson_on_syn_3fold_no_double_cme_1_1.%J.out

python3 learn_richardson_on_syn_3fold_no_double_cme.py 1 1



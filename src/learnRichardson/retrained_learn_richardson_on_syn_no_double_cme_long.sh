#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_retrained_learn_richardson_on_syn_no_double_cme_long.%J.err 

#SBATCH --output=slurm_retrained_learn_richardson_on_syn_no_double_cme_long.%J.out

python3 retrained_learn_richardson_on_syn_no_double_cme_long.py



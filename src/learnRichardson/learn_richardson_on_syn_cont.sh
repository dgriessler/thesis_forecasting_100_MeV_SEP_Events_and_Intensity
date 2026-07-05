#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_learn_richardson_on_syn.%J.err 

#SBATCH --output=slurm_learn_richardson_on_syn.%J.out

python3 learn_richardson_on_syn_cont.py



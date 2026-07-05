#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=25GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_retrained_learn_richardson_syn.%J.err 

#SBATCH --output=slurm_retrained_learn_richardson_syn.%J.out

python3 retrained_learn_richardson_syn.py



#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_rRT_diff_0_9.%J.err 

#SBATCH --output=slurm_rRT_diff_0_9.%J.out

python3 rRT_diff.py 9



#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_rm_rRT_0_7.%J.err 

#SBATCH --output=slurm_rm_rRT_0_7.%J.out

python3 richardson_mixed_rRT.py 7 15000



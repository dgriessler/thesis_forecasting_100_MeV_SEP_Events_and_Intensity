#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_retrained_rRT_0_9_4.%J.err 

#SBATCH --output=slurm_retrained_rRT_0_9_4.%J.out

python3 retrained_rRT_single.py 9 13218 4



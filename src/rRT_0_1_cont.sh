#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=5GB

#SBATCH --time=08:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=4

#SBATCH --error=slurm_rRT_0_1.%J.err 

#SBATCH --output=slurm_rRT_0_1.%J.out

python3 rRT_cont.py 1



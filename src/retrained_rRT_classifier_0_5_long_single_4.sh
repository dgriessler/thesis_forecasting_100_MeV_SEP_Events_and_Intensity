#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=16

#SBATCH --error=slurm_rRT_classifier_0_5_long_4.%J.err 

#SBATCH --output=slurm_rRT_classifier_0_5_long_4.%J.out

python3 retrained_rRT_classifier_long_single.py 5 38486 4



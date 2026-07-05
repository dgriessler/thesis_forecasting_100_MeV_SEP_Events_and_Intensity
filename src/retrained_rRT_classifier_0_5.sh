#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_retrained_rRT_classifier_0_5.%J.err 

#SBATCH --output=slurm_retrained_rRT_classifier_0_5.%J.out

python3 retrained_rRT_classifier.py 5 470



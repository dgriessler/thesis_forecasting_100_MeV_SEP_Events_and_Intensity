#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=8

#SBATCH --error=slurm_utr_along_retrained_rRT_0_3.%J.err 

#SBATCH --output=slurm_utr_along_retrained_rRT_0_3.%J.out

python3 using_training_richardson_alongside_retrained_rRT.py 3 1350



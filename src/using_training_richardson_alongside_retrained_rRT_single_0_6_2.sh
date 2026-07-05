#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_utr_along_retrained_rRT_0_6_2.%J.err 

#SBATCH --output=slurm_utr_along_retrained_rRT_0_6_2.%J.out

python3 using_training_richardson_alongside_retrained_rRT_single.py 6 100450 2



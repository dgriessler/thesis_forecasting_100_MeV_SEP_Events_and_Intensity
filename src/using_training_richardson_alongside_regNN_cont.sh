#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_utr_along_reg.%J.err 

#SBATCH --output=slurm_utr_along_reg.%J.out

python3 using_training_richardson_alongside_regNN_cont.py 18000



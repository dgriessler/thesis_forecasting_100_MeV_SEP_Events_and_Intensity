#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_dense_retrained_rRT_0_5_1.%J.err 

#SBATCH --output=slurm_dense_retrained_rRT_0_5_1.%J.out

python3 dense_loss_retrained_rRT_it_single.py 0.5 45532 1




#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_alt_dense_rRT_cont_0_2.%J.err 

#SBATCH --output=slurm_alt_dense_rRT_cont_0_2.%J.out

python3 alt_dense_loss_rRT_cont.py 0.2 100000



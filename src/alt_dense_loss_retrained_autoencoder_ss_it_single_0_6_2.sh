#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_alt_dense_retrained_auto_0_6_2.%J.err 

#SBATCH --output=slurm_alt_dense_retrained_auto_0_6_2.%J.out

python3 alt_dense_loss_retrained_autoencoder_ss_it_single.py 0.6 21122 2




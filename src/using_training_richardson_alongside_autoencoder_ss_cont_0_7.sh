#!/bin/bash

#SBATCH --job-name DG_Regression_NN
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=15GB

#SBATCH --time=1-00:00:00
#SBATCH --partition=long

#SBATCH --cpus-per-task=32

#SBATCH --error=slurm_utr_along_auto_ss_cont_0_7.%J.err 

#SBATCH --output=slurm_utr_along_auto_ss_cont_0_7.%J.out

python3 using_training_richardson_alongside_autoencoder_ss_cont.py 7 20000



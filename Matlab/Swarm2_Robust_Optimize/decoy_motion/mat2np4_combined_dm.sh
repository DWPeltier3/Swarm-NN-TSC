#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/decoy_motion/mat2np_combined_dm_r40k.txt
#SBATCH --partition=beards

. /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

python mat2np4_combined_dm.py
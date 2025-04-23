#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/decoy_number/mat2np_comb_dn1C10_perR_%j.txt
#SBATCH --partition=beards

. /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

# Variables
decoy_motion="perR"

# Run the Python script
python mat2np4_combined_dn.py \
--decoy_motion $decoy_motion 
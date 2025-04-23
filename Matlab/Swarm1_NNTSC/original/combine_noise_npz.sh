#!/bin/bash
#SBATCH --job-name=swarm_combine_noise_dataset
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/Noise/combine_noise_npz_%j.txt
#SBATCH --partition=beards

source /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

python combine_noise_npz.py
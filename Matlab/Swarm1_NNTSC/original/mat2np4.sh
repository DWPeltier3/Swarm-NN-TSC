#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/mat2np%a.txt
#SBATCH --partition=beards
#SBATCH --array=1-3

. /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

# Define the swarm sizes
swarm_sizes=(25 50 75)
# Get the swarm size based on the SLURM array task ID
swarm_size=${swarm_sizes[$SLURM_ARRAY_TASK_ID-1]}

python mat2np4.py --swarm_size $swarm_size
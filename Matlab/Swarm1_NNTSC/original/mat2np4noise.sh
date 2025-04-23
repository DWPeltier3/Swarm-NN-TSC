#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/mat2np_noise%a.txt
#SBATCH --partition=beards
#SBATCH --array=1-50

. /etc/profile

module load lang/miniconda3/23.1.0

source activate swarm

scaling_factor=$(echo "scale=2; ${SLURM_ARRAY_TASK_ID}/100" | bc)
python mat2np4noise.py --scaling_factor $scaling_factor